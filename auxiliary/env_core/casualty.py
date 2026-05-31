"""Casualty model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env_core.grid import Grid


def compute_tick_casualties(grid: "Grid") -> int:
    """
    Per burning cell: a fraction of the remaining population dies based on intensity.
    Casualties are removed from cell.pop so deaths can't exceed total population.

    Rate: ~5% of remaining pop per turn at intensity 5 (max).
    At intensity 3 that's ~3%, at intensity 1 that's ~1%.
    """
    total = 0
    for cell in grid.cells.values():
        if cell.on_fire and cell.pop > 0:
            rate = (cell.intensity / 5.0) * 0.05
            killed = max(1, round(cell.pop * rate))
            killed = min(killed, cell.pop)
            cell.pop -= killed
            total += killed
    return total


def compute_tick_score_delta(
    grid: "Grid",
    casualties: int,
    newly_ignited: int,
) -> int:
    safe_pop = 0
    burning_pop = 0
    non_burning_cells = 0
    burning_cells = 0

    for cell in grid.cells.values():
        if cell.on_fire:
            burning_cells += 1
            burning_pop += cell.pop
        else:
            non_burning_cells += 1
            safe_pop += cell.pop

    delta = (
        safe_pop * 10
        - burning_pop * 10
        + non_burning_cells * 1
        - burning_cells * 5
        - casualties * 100
        - newly_ignited * 1000
    )
    return delta
