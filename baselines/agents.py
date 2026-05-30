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
    Each crew independently targets the highest (intensity * pop) burning cell
    nearest to it. Suppress if already on a burning cell, otherwise step toward
    target. Crews are assigned distinct targets so they don't pile onto one cell.
    """

    def act(self, obs: dict[str, Any]) -> dict[str, Any]:
        commands = []
        cells = {(c["x"], c["y"]): c for c in obs.get("cells", [])}
        burning = sorted(
            (c for c in cells.values() if c["on_fire"]),
            key=lambda c: -(c["intensity"] * c.get("pop", 1) + c["intensity"]),
        )

        assigned: dict[tuple[int, int], int] = {}  # target -> crew_id

        for crew in obs.get("crews", []):
            remaining = crew["actions_remaining"]
            cx, cy = crew["cell"]
            current = cells.get((cx, cy))

            # Pick best unassigned target, breaking ties by distance
            target = None
            for b in burning:
                key = (b["x"], b["y"])
                if key not in assigned:
                    target = b
                    assigned[key] = crew["id"]
                    break
            if target is None and burning:
                target = burning[0]

            for _ in range(remaining):
                if current and current["on_fire"]:
                    commands.append({"crew": crew["id"], "action": "suppress"})
                elif target:
                    step = _step_toward([cx, cy], [target["x"], target["y"]])
                    commands.append({"crew": crew["id"], "action": "move", "to": step})
                    cx, cy = step
                    current = cells.get((cx, cy))

        return {"commands": commands}


def _step_toward(src: list[int], dst: list[int]) -> list[int]:
    sx, sy = src
    dx, dy = dst
    if abs(dx - sx) >= abs(dy - sy):
        return [sx + (1 if dx > sx else -1), sy]
    return [sx, sy + (1 if dy > sy else -1)]
