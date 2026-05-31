"""Casualty and property loss model."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env_core.grid import Grid


def compute_tick_casualties(grid: "Grid") -> int:
    """
    Per burning cell: casualties scale with intensity and unevacuated pop.
    Pop on a cell driven to intensity 0 this tick is already safe (intensity==0, on_fire==False).
    """
    total = 0
    for cell in grid.cells.values():
        if cell.on_fire and cell.pop > 0:
            share = cell.intensity / 5.0
            total += round(cell.pop * share)
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
