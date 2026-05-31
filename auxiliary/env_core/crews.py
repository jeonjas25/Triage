"""Crew state and action effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from env_core.grid import Grid

# Every crew is a single unit that does the same fixed amount of work per action.
SUPPRESS_POWER = 2      # intensity points removed by one suppress action
PREP_POWER = 1          # catchability points removed by one prep action
EVACUATE_CAPACITY = 50  # people moved by one evacuate action


@dataclass
class Crew:
    id: int
    cell: list[int]  # [x, y]
    actions_per_turn: int
    actions_remaining: int = 0

    def __post_init__(self) -> None:
        self.actions_remaining = self.actions_per_turn

    def reset_actions(self) -> None:
        self.actions_remaining = self.actions_per_turn

    def to_obs(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "cell": self.cell,
            "actions_remaining": self.actions_remaining,
        }


def apply_commands(
    commands: list[dict[str, Any]],
    crews: list[Crew],
    grid: "Grid",
) -> int:
    """
    Apply a list of crew commands. Returns the count of illegal actions rejected.
    Each command costs 1 action point.
    """
    crew_map = {c.id: c for c in crews}
    illegal = 0

    for cmd in commands:
        crew_id = cmd.get("crew")
        action = cmd.get("action")
        crew = crew_map.get(crew_id)

        if crew is None or action not in ("move", "suppress", "prep", "evacuate"):
            illegal += 1
            continue

        if crew.actions_remaining <= 0:
            illegal += 1
            continue

        if action == "move":
            to = cmd.get("to")
            if not _is_adjacent(crew.cell, to, grid):
                illegal += 1
                continue
            crew.cell = list(to)
            crew.actions_remaining -= 1

        elif action == "suppress":
            cell = grid.get(crew.cell[0], crew.cell[1])
            if cell is None or not cell.on_fire:
                illegal += 1
                continue
            cell.intensity = max(0, cell.intensity - SUPPRESS_POWER)
            if cell.intensity == 0:
                cell.on_fire = False
            crew.actions_remaining -= 1

        elif action == "prep":
            cell = grid.get(crew.cell[0], crew.cell[1])
            if cell is None:
                illegal += 1
                continue
            cell.catchability = max(0, cell.catchability - PREP_POWER)
            crew.actions_remaining -= 1

        elif action == "evacuate":
            to = cmd.get("to")
            if not _is_adjacent(crew.cell, to, grid):
                illegal += 1
                continue
            src = grid.get(crew.cell[0], crew.cell[1])
            if src is None:
                illegal += 1
                continue
            moved = min(EVACUATE_CAPACITY, src.pop)
            src.pop -= moved
            crew.cell = list(to)
            crew.actions_remaining -= 1

    return illegal


def _is_adjacent(cell_a: list[int], cell_b: Any, grid: "Grid") -> bool:
    if not cell_b or len(cell_b) != 2:
        return False
    dx = abs(cell_a[0] - cell_b[0])
    dy = abs(cell_a[1] - cell_b[1])
    if dx + dy != 1:
        return False
    return grid.get(cell_b[0], cell_b[1]) is not None
