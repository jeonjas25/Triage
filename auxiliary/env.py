"""
Triage benchmark environment.

Implements reset() and step() for the mesocosm platform.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

from bench_common.env_sdk.base import BaseEnv, StepResult
from env_core.casualty import compute_tick_casualties, compute_tick_score_delta
from env_core.crews import Crew, apply_commands
from env_core.fire import FireModel
from env_core.grid import Grid, load_scenario

_SCENARIOS_DIR = Path(__file__).parent.parent / "scenarios"

_BUILTIN_DEFAULT: dict[str, Any] = {
    "name": "two_towns",
    "grid_width": 14, "grid_height": 14, "max_turns": 20, "actions_per_crew": 4,
    "crews": [{"id": 0, "size": 10, "cell": [7, 0]}, {"id": 1, "size": 10, "cell": [7, 13]}],
    "fire_clusters": [
        {"cells": [[3, 2], [2, 3], [3, 3]], "intensity": 40},
        {"cells": [[10, 11], [11, 10], [10, 10]], "intensity": 40},
    ],
    "cells": [
        {"x": 2,  "y": 2,  "pop": 300, "property_value": 3000, "spreadability": 65},
        {"x": 1,  "y": 2,  "pop": 50,  "property_value": 500,  "spreadability": 60},
        {"x": 2,  "y": 1,  "pop": 50,  "property_value": 500,  "spreadability": 60},
        {"x": 11, "y": 11, "pop": 250, "property_value": 2500, "spreadability": 65},
        {"x": 12, "y": 11, "pop": 40,  "property_value": 400,  "spreadability": 60},
        {"x": 11, "y": 12, "pop": 40,  "property_value": 400,  "spreadability": 60},
    ],
    "baseline_donothing": -582038,
    "baseline_greedy": -464967,
}


def _load_named_scenario(name: str) -> dict[str, Any]:
    path = _SCENARIOS_DIR / f"{name}.json"
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        if name == "two_towns":
            return _BUILTIN_DEFAULT
        raise


def _default_scenario() -> dict[str, Any]:
    try:
        return _load_named_scenario("two_towns")
    except Exception:
        return _BUILTIN_DEFAULT


class TriageEnv(BaseEnv):
    def __init__(self) -> None:
        self._grid: Grid | None = None
        self._crews: list[Crew] = []
        self._fire = FireModel()
        self._schedule: list[dict] = []
        self._burn_lifespans: dict[tuple[int, int], int] = {}
        self._active_timers: dict[tuple[int, int], int] = {}
        self._pending_clusters: list[tuple[int, dict]] = []  # (ignite_at_turn, cluster)
        self._turn: int = 0
        self._max_turns: int = 20
        self._running_total: int = 0
        self._casualties: int = 0
        self._property_lost: int = 0
        self._cells_ignited: int = 0
        self._illegal: int = 0
        self._baseline_donothing: int = 0
        self._baseline_greedy: int = 1
        self._scenario: dict[str, Any] = _default_scenario()

    def reset(self, seed: int | None = None, **params: Any) -> dict[str, Any]:
        rng = random.Random(seed)
        raw = params.get("scenario", self._scenario)
        scenario = _load_named_scenario(raw) if isinstance(raw, str) else raw

        self._grid, clusters = load_scenario(scenario)
        self._max_turns = scenario.get("max_turns", 20)
        self._baseline_donothing = scenario.get("baseline_donothing", -50000)
        self._baseline_greedy = scenario.get("baseline_greedy", 10000)
        actions_per_crew = scenario.get("actions_per_crew", 4)

        self._crews = [
            Crew(
                id=c["id"],
                size=c["size"],
                cell=list(c["cell"]),
                actions_per_turn=actions_per_crew,
            )
            for c in scenario.get("crews", [])
        ]

        assert self._grid is not None
        self._pending_clusters = []
        for cluster in clusters:
            ignite_at = cluster.get("ignite_at", 0)
            if ignite_at <= 0:
                for cx, cy in cluster["cells"]:
                    cell = self._grid.get(cx, cy)
                    if cell:
                        cell.on_fire = True
                        cell.intensity = cluster.get("intensity", 20)
            else:
                self._pending_clusters.append((ignite_at, cluster))

        self._schedule, self._burn_lifespans = self._preroll(rng, self._grid, self._max_turns)

        self._active_timers = {}
        for cell in self._grid.cells.values():
            if cell.on_fire:
                self._active_timers[(cell.x, cell.y)] = self._burn_lifespans[(cell.x, cell.y)]

        self._turn = 0
        self._running_total = 0
        self._casualties = 0
        self._property_lost = 0
        self._cells_ignited = 0
        self._illegal = 0

        return self._observation()

    def step(self, action: Any) -> StepResult:
        assert self._grid is not None

        # Ignite any clusters whose turn has come
        still_pending = []
        for ignite_at, cluster in self._pending_clusters:
            if self._turn >= ignite_at:
                for cx, cy in cluster["cells"]:
                    cell = self._grid.get(cx, cy)
                    if cell and not cell.on_fire:
                        cell.on_fire = True
                        cell.intensity = cluster.get("intensity", 20)
                        key = (cx, cy)
                        self._active_timers[key] = self._burn_lifespans.get(key, 15)
            else:
                still_pending.append((ignite_at, cluster))
        self._pending_clusters = still_pending

        commands = action.get("commands", []) if isinstance(action, dict) else []
        illegal = apply_commands(commands, self._crews, self._grid)
        self._illegal += illegal

        schedule_t = self._schedule[self._turn]
        newly_ignited, prop_lost = self._fire.step(
            self._grid, schedule_t, self._active_timers, self._burn_lifespans
        )
        casualties = compute_tick_casualties(self._grid)
        delta = compute_tick_score_delta(
            self._grid, casualties, newly_ignited, prop_lost
        )

        self._running_total += delta
        self._casualties += casualties
        self._property_lost += prop_lost
        self._cells_ignited += newly_ignited
        self._turn += 1

        for crew in self._crews:
            crew.reset_actions()

        terminated = (
            not any(c.on_fire for c in self._grid.cells.values())
            and not self._pending_clusters
        )
        truncated = self._turn >= self._max_turns

        info: dict[str, Any] = {}
        if terminated or truncated:
            denom = max(1, self._baseline_greedy - self._baseline_donothing)
            norm = (self._running_total - self._baseline_donothing) / denom
            info = {
                "norm_score": round(norm, 4),
                "lives_lost": self._casualties,
                "property_lost": self._property_lost,
                "cells_burned": self._cells_ignited,
                "illegal_actions": self._illegal,
            }

        reward = float(delta)
        return StepResult(
            observation=self._observation(),
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _observation(self) -> dict[str, Any]:
        assert self._grid is not None
        relevant = self._grid.observation_cells(self._crews)
        return {
            "turn": self._turn,
            "max_turns": self._max_turns,
            "grid_width": self._grid.width,
            "grid_height": self._grid.height,
            "score": self._running_total,
            "crews": [c.to_obs() for c in self._crews],
            "cells": [
                {
                    "x": cell.x,
                    "y": cell.y,
                    "pop": cell.pop,
                    "on_fire": cell.on_fire,
                    "intensity": cell.intensity,
                    "spreadability": cell.spreadability,
                    "property_value": cell.property_value,
                    "property_remaining": cell.property_remaining,
                }
                for cell in relevant
            ],
        }

    @staticmethod
    def _preroll(rng: random.Random, grid: Grid, max_turns: int) -> tuple[list[dict], dict]:
        burn_lifespans = {(c.x, c.y): rng.randint(12, 20) for c in grid.cells.values()}
        schedule = []
        for _ in range(max_turns):
            schedule.append({
                "ignition_rolls": {(c.x, c.y): rng.random() for c in grid.cells.values()},
                "intensity_growth": {(c.x, c.y): rng.randint(2, 8) for c in grid.cells.values()},
            })
        return schedule, burn_lifespans
