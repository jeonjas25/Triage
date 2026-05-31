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
    ) -> int:
        """
        Advance every burning cell one tick.
        active_timers is mutated in place.
        Returns newly_ignited_count.

        Spread uses combined probability across all burning neighbors:
            P(ignite) = 1 - prod(1 - p_i)  for each burning neighbor i
            p_i = (target.catchability/5) * (source.intensity/100)
        A single pre-rolled value per cell is compared against P(ignite),
        so the outcome is deterministic for identical seeds regardless of
        how many neighbors are burning or in what order they are visited.
        """
        newly_ignited = 0

        burning = [c for c in grid.cells.values() if c.on_fire]

        # 1. Grow intensity on burning cells
        for cell in burning:
            increment = schedule_t["intensity_growth"].get((cell.x, cell.y), 0)
            cell.intensity = min(100, cell.intensity + increment)

        # 2. Tick burn timers; extinguish cells whose timer hits zero
        for cell in burning:
            key = (cell.x, cell.y)
            timer = active_timers.get(key, 1)
            if timer <= 1:
                cell.on_fire = False
                cell.intensity = 0
                active_timers.pop(key, None)
            else:
                active_timers[key] = timer - 1

        # 3. Accumulate ignition pressure on non-burning neighbors from ALL
        #    still-burning sources. Combined probability uses the complement
        #    product so multiple burning neighbors compound correctly.
        p_no_ignite: dict[tuple[int, int], float] = {}
        for src in [c for c in grid.cells.values() if c.on_fire]:
            for nb in grid.neighbors(src.x, src.y):
                if nb.on_fire:
                    continue
                key = (nb.x, nb.y)
                p_src = (nb.catchability / 5.0) * (src.intensity / 100.0)
                if key not in p_no_ignite:
                    p_no_ignite[key] = 1.0
                p_no_ignite[key] *= (1.0 - p_src)

        # 4. Roll once per candidate using the pre-rolled schedule value
        for key, p_no in p_no_ignite.items():
            p_ignite = 1.0 - p_no
            roll = schedule_t["ignition_rolls"].get(key, 1.0)
            if roll < p_ignite:
                nb = grid.get(key[0], key[1])
                if nb and not nb.on_fire:
                    nb.on_fire = True
                    nb.intensity = 1
                    active_timers[key] = burn_lifespans.get(key, 15)
                    newly_ignited += 1

        return newly_ignited

    def intensity(self, cell) -> float:
        return cell.intensity / 5.0
