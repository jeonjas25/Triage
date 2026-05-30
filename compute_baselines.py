"""
Compute do-nothing and greedy baseline running totals for a set of scenarios.
Run this to get numbers to bake into scenario configs.

Usage:
    python compute_baselines.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "auxiliary"))
sys.path.insert(0, str(Path(__file__).parent))

from env import TriageEnv
from baselines.agents import DoNothingAgent, GreedyAgent


def run_raw(scenario: dict, agent, seed: int) -> int:
    """Run one episode and return the raw _running_total (before normalization)."""
    env = TriageEnv()
    # Patch baselines so normalization doesn't interfere with raw read
    env._baseline_donothing = 0
    env._baseline_greedy = 1
    obs = env.reset(seed=seed, scenario=scenario)
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        if result.terminated or result.truncated:
            return env._running_total


def compute_for_scenario(name: str, scenario: dict, seeds: list[int]) -> None:
    print(f"\n=== {name} ===")
    for seed in seeds:
        dn = run_raw(scenario, DoNothingAgent(), seed)
        gr = run_raw(scenario, GreedyAgent(), seed)
        print(f"  seed={seed}  do_nothing={dn:>10,}  greedy={gr:>10,}  spread={gr - dn:>10,}")


SCENARIOS: dict[str, dict] = {
    # Two towns, two fire fronts igniting right next to each — crews can hold one line.
    "two_towns": {
        "grid_width": 14,
        "grid_height": 14,
        "max_turns": 20,
        "actions_per_crew": 4,
        "crews": [
            {"id": 0, "size": 10, "cell": [7, 0]},
            {"id": 1, "size": 10, "cell": [7, 13]},
        ],
        "fire_clusters": [
            # Front A: burning cells directly adjacent to Town A (2,2)
            {"cells": [[3, 2], [2, 3], [3, 3]], "intensity": 40},
            # Front B: burning cells directly adjacent to Town B (11,11)
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
        "baseline_donothing": 0,
        "baseline_greedy": 1,
    },
    # Three fronts, two crews — must visibly abandon one front to save the others.
    "three_fronts": {
        "grid_width": 18,
        "grid_height": 18,
        "max_turns": 25,
        "actions_per_crew": 4,
        "crews": [
            {"id": 0, "size": 10, "cell": [9, 0]},
            {"id": 1, "size": 10, "cell": [9, 17]},
        ],
        "fire_clusters": [
            # Front A: top-left town (2,2)
            {"cells": [[3, 2], [2, 3], [3, 3]], "intensity": 35},
            # Front B: top-right town (15,2)
            {"cells": [[14, 2], [15, 3], [14, 3]], "intensity": 35},
            # Front C: bottom-center town (9,15)
            {"cells": [[9, 14], [8, 14], [10, 14]], "intensity": 25},
        ],
        "cells": [
            {"x": 2,  "y": 2,  "pop": 400, "property_value": 4000, "spreadability": 70},
            {"x": 1,  "y": 2,  "pop": 60,  "property_value": 600,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 60,  "property_value": 600,  "spreadability": 60},
            {"x": 15, "y": 2,  "pop": 350, "property_value": 3500, "spreadability": 70},
            {"x": 16, "y": 2,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 15, "y": 1,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 9,  "y": 15, "pop": 200, "property_value": 2000, "spreadability": 60},
            {"x": 8,  "y": 15, "pop": 30,  "property_value": 300,  "spreadability": 55},
            {"x": 10, "y": 15, "pop": 30,  "property_value": 300,  "spreadability": 55},
        ],
        "baseline_donothing": 0,
        "baseline_greedy": 1,
    },
    # Squeeze: small crews, fast-spreading fires, forced triage.
    "squeeze": {
        "grid_width": 16,
        "grid_height": 16,
        "max_turns": 22,
        "actions_per_crew": 3,
        "crews": [
            {"id": 0, "size": 7, "cell": [8, 0]},
            {"id": 1, "size": 7, "cell": [8, 15]},
        ],
        "fire_clusters": [
            # Front A: fires right on the border of Town A (1,1)
            {"cells": [[2, 1], [1, 2], [2, 2]], "intensity": 45},
            # Front B: fires right on the border of Town B (14,1)
            {"cells": [[13, 1], [14, 2], [13, 2]], "intensity": 45},
            # Front C: fires threatening Town C (8,14)
            {"cells": [[8, 13], [7, 13], [9, 13]], "intensity": 30},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 500, "property_value": 5000, "spreadability": 75},
            {"x": 1,  "y": 2,  "pop": 80,  "property_value": 800,  "spreadability": 65},
            {"x": 2,  "y": 1,  "pop": 80,  "property_value": 800,  "spreadability": 65},
            {"x": 14, "y": 1,  "pop": 450, "property_value": 4500, "spreadability": 75},
            {"x": 14, "y": 2,  "pop": 70,  "property_value": 700,  "spreadability": 65},
            {"x": 13, "y": 1,  "pop": 70,  "property_value": 700,  "spreadability": 65},
            {"x": 8,  "y": 14, "pop": 250, "property_value": 2500, "spreadability": 65},
            {"x": 7,  "y": 14, "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 9,  "y": 14, "pop": 40,  "property_value": 400,  "spreadability": 60},
        ],
        "baseline_donothing": 0,
        "baseline_greedy": 1,
    },
}


if __name__ == "__main__":
    seeds = [42, 43, 44]
    for name, scenario in SCENARIOS.items():
        compute_for_scenario(name, scenario, seeds)

    print("\nPaste the seed=42 values into each scenario config as baseline_donothing / baseline_greedy.")
