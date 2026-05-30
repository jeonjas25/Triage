"""Wildfire cellular automaton."""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from env_core.grid import Grid


class FireModel:
    def step(
        self,
        grid: "Grid",
        schedule_t: dict,
        active_timers: dict[tuple[int, int], int],
        burn_lifespans: dict[tuple[int, int], int],
    ) -> tuple[int, int]:
        """
        Advance every burning cell one tick.
        active_timers is mutated in place.
        Returns (newly_ignited_count, property_lost_this_tick).
        """
        newly_ignited = 0
        property_lost = 0

        burning = [c for c in grid.cells.values() if c.on_fire]

        # Grow intensity on existing burning cells
        for cell in burning:
            increment = schedule_t["intensity_growth"].get((cell.x, cell.y), 0)
            cell.intensity = min(100, cell.intensity + increment)

            if cell.intensity > cell.peak_intensity:
                delta_peak = cell.intensity - cell.peak_intensity
                cell.peak_intensity = cell.intensity
                prop_delta = cell.property_value * delta_peak // 100
                cell.property_remaining = max(0, cell.property_remaining - prop_delta)
                property_lost += prop_delta

        # Tick burn timers; burn out cells at zero
        for cell in burning:
            key = (cell.x, cell.y)
            timer = active_timers.get(key, 1)
            if timer <= 1:
                cell.on_fire = False
                cell.intensity = 0
                active_timers.pop(key, None)
            else:
                active_timers[key] = timer - 1

        # Spread to neighbors (only from still-burning cells)
        for cell in [c for c in grid.cells.values() if c.on_fire]:
            for nb in grid.neighbors(cell.x, cell.y):
                if nb.on_fire:
                    continue
                roll = schedule_t["ignition_rolls"].get((nb.x, nb.y), 1.0)
                likelihood = (nb.spreadability / 100.0) * (cell.intensity / 100.0)
                if roll < likelihood:
                    nb.on_fire = True
                    nb.intensity = 1
                    key = (nb.x, nb.y)
                    active_timers[key] = burn_lifespans.get(key, 15)
                    newly_ignited += 1

        return newly_ignited, property_lost

    def intensity(self, cell) -> float:
        return cell.intensity / 100.0
