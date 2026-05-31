"""Random and greedy baseline agents (local use only, not shipped to mesocosm)."""

from __future__ import annotations

import random
from typing import Any


class DoNothingAgent:
    """Submits no commands every turn."""

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        return {"commands": []}


class RandomAgent:
    """Issues random legal-looking commands for each crew."""

    def __init__(self, seed: int | None = None) -> None:
        self._rng = random.Random(seed)

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        commands = []
        directions = [[1, 0], [-1, 0], [0, 1], [0, -1]]
        actions = ["move", "suppress", "prep", "evacuate"]

        for crew in obs.get("crews", []):
            remaining = crew["actions_remaining"]
            cx, cy = crew["cell"]
            for _ in range(remaining):
                action = self._rng.choice(actions)
                if action in ("move", "evacuate"):
                    dx, dy = self._rng.choice(directions)
                    commands.append({"crew": crew["id"], "action": action, "to": [cx + dx, cy + dy]})
                else:
                    commands.append({"crew": crew["id"], "action": action})

        return {"commands": commands}


class GreedyAgent:
    """
    Heuristic agent that suppresses, evacuates, and preps based on urgency.

    Priority per action:
      1. Evacuate — if standing on a burning cell with significant population,
         carry people to the nearest non-burning neighbor. Each evacuate action
         moves a fixed number of people immediately.
      2. Suppress — if standing on or adjacent to a burning cell, reduce intensity.
         Prefer the burning cell with the most population at risk.
      3. Prep — if standing on a non-burning cell that is adjacent to fire and has
         high catchability, reduce fuel load to slow the advance.
      4. Move — step toward the highest-value unaddressed fire front.

    Crews are given distinct target fires so they don't all pile onto one cell.
    """

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        commands: list[dict] = []
        cells = {(c["x"], c["y"]): c for c in obs.get("cells", [])}

        def on_fire(c):
            return bool(c.get("fire") or c.get("on_fire"))

        def catchability(c):
            return c.get("catch", c.get("spread", c.get("catchability", 3)))

        # Rank fires by (intensity * pop_at_risk + intensity) descending
        burning = sorted(
            (c for c in cells.values() if on_fire(c)),
            key=lambda c: -(c.get("intensity", 0) * (c.get("pop", 0) + 1)),
        )

        assigned: dict[tuple[int, int], int] = {}  # fire_key -> crew_id

        for crew in obs.get("crews", []):
            remaining = crew["actions_remaining"]
            cx, cy = crew["cell"]

            # Assign this crew a target fire (prefer uncontested targets)
            target = None
            for b in burning:
                key = (b["x"], b["y"])
                if key not in assigned:
                    target = b
                    assigned[key] = crew["id"]
                    break
            if target is None and burning:
                target = burning[0]

            grid = obs.get("grid", [obs.get("grid_width", 14), obs.get("grid_height", 14)])
            for _ in range(remaining):
                current = cells.get((cx, cy))
                cmd = _best_action(crew["id"], cx, cy, current, target, cells, grid[0], grid[1], on_fire, catchability)
                commands.append(cmd)
                # Track position changes so subsequent actions in this turn are correct
                if cmd["action"] in ("move", "evacuate") and "to" in cmd:
                    cx, cy = cmd["to"]

        return {"commands": commands}


def _best_action(
    crew_id: int,
    cx: int, cy: int,
    current: dict | None,
    target: dict | None,
    cells: dict,
    W: int, H: int,
    on_fire=None,
    catchability=None,
) -> dict:
    if on_fire is None:
        on_fire = lambda c: bool(c.get("fire") or c.get("on_fire"))
    if catchability is None:
        catchability = lambda c: c.get("catch", 3)
    dirs = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    # 1. EVACUATE — standing on burning cell with meaningful population
    if current and on_fire(current) and current.get("pop", 0) > 30:
        safe = _safest_neighbor(cx, cy, cells, dirs, W, H, on_fire)
        if safe:
            return {"crew": crew_id, "action": "evacuate", "to": list(safe)}

    # 2. SUPPRESS — standing on burning cell
    if current and on_fire(current):
        return {"crew": crew_id, "action": "suppress"}

    # 3. Find best adjacent burning cell
    best_nb_fire: dict | None = None
    for dx, dy in dirs:
        nx, ny = cx + dx, cy + dy
        nb = cells.get((nx, ny))
        if nb and on_fire(nb) and (
            best_nb_fire is None
            or nb.get("intensity", 0) * (nb.get("pop", 0) + 1) > best_nb_fire.get("intensity", 0) * (best_nb_fire.get("pop", 0) + 1)
        ):
            best_nb_fire = nb

    # 4. PREP — adjacent to fire, on a high-catchability cell with population at risk
    if current and not on_fire(current):
        if best_nb_fire is not None and catchability(current) >= 4 and current.get("pop", 0) > 50:
            return {"crew": crew_id, "action": "prep"}

    # 5. MOVE toward target
    if target:
        dest = [target["x"], target["y"]]
        if [cx, cy] == dest:
            return {"crew": crew_id, "action": "suppress"}
        step = _step_toward([cx, cy], dest)
        return {"crew": crew_id, "action": "move", "to": step}

    # 6. Truly idle
    if current and on_fire(current):
        return {"crew": crew_id, "action": "suppress"}
    return {"crew": crew_id, "action": "prep"}


def _safest_neighbor(
    cx: int, cy: int,
    cells: dict,
    dirs: list,
    W: int, H: int,
    on_fire=None,
) -> tuple[int, int] | None:
    """Return the adjacent cell that is not on fire and has the lowest catchability."""
    if on_fire is None:
        on_fire = lambda c: bool(c.get("fire") or c.get("on_fire"))
    candidates = []
    for dx, dy in dirs:
        nx, ny = cx + dx, cy + dy
        if not (0 <= nx < W and 0 <= ny < H):
            continue
        nb = cells.get((nx, ny))
        if nb is None or not on_fire(nb):
            cat = nb.get("catch", 3) if nb else 3
            candidates.append((cat, nx, ny))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1], candidates[0][2]


def _step_toward(src: list[int], dst: list[int]) -> list[int]:
    sx, sy = src
    dx, dy = dst
    if abs(dx - sx) >= abs(dy - sy):
        return [sx + (1 if dx > sx else -1), sy]
    return [sx, sy + (1 if dy > sy else -1)]
