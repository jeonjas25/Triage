"""Crew state and action effects."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from env_core.grid import Grid

# Every crew is a single unit that does the same fixed amount of work per action.
SUPPRESS_POWER = 3      # intensity points removed per crew member per suppress action
PREP_POWER = 1          # catchability points removed by one prep action
EVACUATE_CAPACITY = 50  # people moved by one evacuate action


@dataclass
class Crew:
    id: int
    size: int
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
            "size": self.size,
            "cell": self.cell,
            "actions_remaining": self.actions_remaining,
        }


def apply_commands(
    commands: list[dict[str, Any]],
    crews: list[Crew],
    grid: "Grid",
    next_crew_id_box: list[int] | None = None,
) -> tuple[int, list[Crew]]:
    """
    Apply a list of crew commands.
    Returns (illegal_count, new_crews_spawned_by_split).

    Actions:
      move      - move to adjacent cell [x,y]
      suppress  - reduce intensity of current burning cell by crew_size*3
      prep      - reduce spreadability of current cell by crew_size*3
      evacuate  - carry up to crew_size*5 people to adjacent cell [x,y]; crew moves with them
      split     - detach `size` members into a new crew at the same cell;
                  the new crew gets full actions immediately.
                  Requires {"crew": id, "action": "split", "size": N}
                  where 0 < N < crew.size.
    """
    crew_map = {c.id: c for c in crews}
    illegal = 0
    new_crews: list[Crew] = []

    for cmd in commands:
        crew_id = cmd.get("crew")
        action  = cmd.get("action")
        crew    = crew_map.get(crew_id)

        if crew is None or action not in ("move", "suppress", "prep", "evacuate", "split"):
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
            cell.intensity = max(0, cell.intensity - crew.size * SUPPRESS_POWER)
            if cell.intensity == 0:
                cell.on_fire = False
            crew.actions_remaining -= 1

        elif action == "prep":
            cell = grid.get(crew.cell[0], crew.cell[1])
            if cell is None:
                illegal += 1
                continue
            cell.catchability = max(1, cell.catchability - PREP_POWER)
            crew.actions_remaining -= 1

        elif action == "evacuate":
            to = cmd.get("to")
            if not _is_adjacent(crew.cell, to, grid):
                illegal += 1
                continue
            src = grid.get(crew.cell[0], crew.cell[1])
            dst = grid.get(to[0], to[1])
            if src is None or dst is None:
                illegal += 1
                continue
            moved = min(EVACUATE_CAPACITY, src.pop)
            src.pop -= moved
            dst.pop += moved
            crew.cell = list(to)
            crew.actions_remaining -= 1

        elif action == "split":
            split_size = cmd.get("size")
            if (
                not isinstance(split_size, int)
                or split_size <= 0
                or split_size >= crew.size
            ):
                illegal += 1
                continue
            if next_crew_id_box is not None:
                new_id = next_crew_id_box[0]
                next_crew_id_box[0] += 1
            else:
                # Fallback: use max existing id + 1 + offset
                all_ids = [c.id for c in crews] + [c.id for c in new_crews]
                new_id = max(all_ids) + 1
            new_crew = Crew(
                id=new_id,
                size=split_size,
                cell=list(crew.cell),
                actions_per_turn=crew.actions_per_turn,
                actions_remaining=crew.actions_per_turn,
            )
            crew_map[new_id] = new_crew  # visible to subsequent commands this turn
            new_crews.append(new_crew)
            crew.size -= split_size
            crew.actions_remaining -= 1

    return illegal, new_crews


def _is_adjacent(cell_a: list[int], cell_b: Any, grid: "Grid") -> bool:
    if not cell_b or len(cell_b) != 2:
        return False
    dx = abs(cell_a[0] - cell_b[0])
    dy = abs(cell_a[1] - cell_b[1])
    if dx + dy != 1:
        return False
    return grid.get(cell_b[0], cell_b[1]) is not None
