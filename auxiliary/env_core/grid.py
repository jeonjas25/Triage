"""Grid and scenario generator."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class Cell:
    x: int
    y: int
    pop: int = 0
    property_value: int = 0
    spreadability: int = 50
    on_fire: bool = False
    intensity: int = 0
    property_remaining: int = 0
    peak_intensity: int = 0

    def __post_init__(self) -> None:
        if self.property_remaining == 0:
            self.property_remaining = self.property_value


@dataclass
class Grid:
    width: int
    height: int
    cells: dict[tuple[int, int], Cell] = field(default_factory=dict)

    def __post_init__(self) -> None:
        for y in range(self.height):
            for x in range(self.width):
                self.cells[(x, y)] = Cell(x=x, y=y)

    def get(self, x: int, y: int) -> Cell | None:
        return self.cells.get((x, y))

    def neighbors(self, x: int, y: int) -> list[Cell]:
        result = []
        for dx, dy in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            c = self.get(x + dx, y + dy)
            if c is not None:
                result.append(c)
        return result

    def observation_cells(self, crews: list[Any]) -> list[Cell]:
        """Return only decision-relevant cells."""
        relevant: set[tuple[int, int]] = set()
        crew_cells = {tuple(cr.cell) for cr in crews}

        for cell in self.cells.values():
            key = (cell.x, cell.y)
            if cell.on_fire or cell.pop > 0 or key in crew_cells:
                relevant.add(key)
                for nb in self.neighbors(cell.x, cell.y):
                    relevant.add((nb.x, nb.y))

        return [self.cells[k] for k in relevant]


def load_scenario(config: dict[str, Any]) -> tuple[Grid, list[dict[str, Any]]]:
    """Build a Grid and list of initial fire cluster specs from a scenario config dict."""
    grid = Grid(width=config["grid_width"], height=config["grid_height"])

    for cp in config.get("cells", []):
        cell = grid.get(cp["x"], cp["y"])
        if cell is None:
            continue
        cell.pop = cp.get("pop", 0)
        cell.property_value = cp.get("property_value", 0)
        cell.property_remaining = cell.property_value
        cell.spreadability = cp.get("spreadability", 50)

    return grid, config.get("fire_clusters", [])
