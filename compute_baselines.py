"""
Compute do-nothing and greedy baseline running totals for all scenarios.
Run this whenever scenarios change to get numbers to bake into JSON configs.

Usage:
    python compute_baselines.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent / "auxiliary"))
sys.path.insert(0, str(Path(__file__).parent))

from env import TriageEnv
from baselines.agents import DoNothingAgent, GreedyAgent


def run_raw(scenario: dict, agent, seed: int) -> int:
    env = TriageEnv()
    env._baseline_donothing = 0
    env._baseline_greedy = 1
    obs = env.reset(seed=seed, scenario=scenario)
    while True:
        result = env.step(agent.act(obs))
        obs = result.observation
        if result.terminated or result.truncated:
            return env._running_total


SCENARIOS: dict[str, dict] = {

    # ── EXISTING (keep baselines stable) ────────────────────────────────
    "two_towns": {
        "description": "Two towns, two adjacent fronts. Crews can hold one line.",
        "grid_width": 14, "grid_height": 14, "max_turns": 20, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [7, 0]}, {"id": 1, "size": 10, "cell": [7, 13]}],
        "fire_clusters": [
            {"cells": [[3, 2], [2, 3], [3, 3]], "intensity": 40},
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
    },

    "three_fronts": {
        "description": "Three fronts, two crews. Must abandon one town.",
        "grid_width": 18, "grid_height": 18, "max_turns": 25, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [9, 0]}, {"id": 1, "size": 10, "cell": [9, 17]}],
        "fire_clusters": [
            {"cells": [[3, 2], [2, 3], [3, 3]], "intensity": 35},
            {"cells": [[14, 2], [15, 3], [14, 3]], "intensity": 35},
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
    },

    "squeeze": {
        "description": "Three fast fronts, small crews with 3 AP. Hardest budget.",
        "grid_width": 16, "grid_height": 16, "max_turns": 22, "actions_per_crew": 3,
        "crews": [{"id": 0, "size": 7, "cell": [8, 0]}, {"id": 1, "size": 7, "cell": [8, 15]}],
        "fire_clusters": [
            {"cells": [[2, 1], [1, 2], [2, 2]], "intensity": 45},
            {"cells": [[13, 1], [14, 2], [13, 2]], "intensity": 45},
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
    },

    # ── NEW SCENARIOS ────────────────────────────────────────────────────

    # Asymmetric stakes: north 6x more valuable. Correct play: protect north.
    "asymmetric_stakes": {
        "description": "Two fronts, north town has 6x the population. Priority is obvious but execution is hard.",
        "grid_width": 14, "grid_height": 14, "max_turns": 20, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [7, 6]}, {"id": 1, "size": 8, "cell": [7, 7]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3]], "intensity": 38},
            {"cells": [[11, 11], [10, 11], [11, 10]], "intensity": 38},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 600, "property_value": 6000, "spreadability": 65},
            {"x": 1,  "y": 2,  "pop": 80,  "property_value": 800,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 80,  "property_value": 800,  "spreadability": 60},
            {"x": 12, "y": 12, "pop": 80,  "property_value": 800,  "spreadability": 60},
            {"x": 12, "y": 11, "pop": 15,  "property_value": 150,  "spreadability": 55},
            {"x": 11, "y": 12, "pop": 15,  "property_value": 150,  "spreadability": 55},
        ],
    },

    # Fast vs slow: one inferno, one smolder. Timing changes the optimal play.
    "racing_fronts": {
        "description": "North fire is high-intensity on dry terrain; south smolders slowly. Equal populations.",
        "grid_width": 16, "grid_height": 12, "max_turns": 18, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [8, 6]}, {"id": 1, "size": 10, "cell": [7, 5]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3], [3, 3]], "intensity": 55},
            {"cells": [[13, 8], [12, 8], [13, 7]], "intensity": 18},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 350, "property_value": 3500, "spreadability": 80},
            {"x": 1,  "y": 2,  "pop": 50,  "property_value": 500,  "spreadability": 75},
            {"x": 2,  "y": 1,  "pop": 50,  "property_value": 500,  "spreadability": 75},
            {"x": 14, "y": 9,  "pop": 300, "property_value": 3000, "spreadability": 40},
            {"x": 14, "y": 10, "pop": 40,  "property_value": 400,  "spreadability": 35},
            {"x": 15, "y": 9,  "pop": 40,  "property_value": 400,  "spreadability": 35},
        ],
    },

    # Delayed reinforcement: third fire ignites at turn 6. Dynamic reallocation required.
    "delayed_storm": {
        "description": "Two fires at start, a third ignites at turn 6. Tests dynamic crew reallocation.",
        "grid_width": 18, "grid_height": 18, "max_turns": 25, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [9, 0]}, {"id": 1, "size": 10, "cell": [0, 9]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3]], "intensity": 35},
            {"cells": [[15, 15], [14, 15], [15, 14]], "intensity": 35},
            {"cells": [[15, 2], [14, 2], [15, 3]], "intensity": 40, "ignite_at": 6},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 350, "property_value": 3500, "spreadability": 65},
            {"x": 1,  "y": 2,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 16, "y": 16, "pop": 300, "property_value": 3000, "spreadability": 65},
            {"x": 16, "y": 15, "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 15, "y": 16, "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 16, "y": 1,  "pop": 280, "property_value": 2800, "spreadability": 68},
            {"x": 16, "y": 2,  "pop": 40,  "property_value": 400,  "spreadability": 62},
            {"x": 17, "y": 1,  "pop": 40,  "property_value": 400,  "spreadability": 62},
        ],
    },

    # One crew, two equal fronts: hardest triage.
    "one_crew": {
        "description": "Single large crew, two equal fire fronts. Must pick one and commit.",
        "grid_width": 12, "grid_height": 12, "max_turns": 20, "actions_per_crew": 5,
        "crews": [{"id": 0, "size": 15, "cell": [6, 6]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3]], "intensity": 40},
            {"cells": [[9, 9], [8, 9], [9, 8]], "intensity": 40},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 300, "property_value": 3000, "spreadability": 65},
            {"x": 1,  "y": 2,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 10, "y": 10, "pop": 250, "property_value": 2500, "spreadability": 65},
            {"x": 10, "y": 9,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 9,  "y": 10, "pop": 40,  "property_value": 400,  "spreadability": 60},
        ],
    },

    # Three unequal: 500 / 150 / 30 pop. Optimal strategy is clear; execution is not.
    "three_unequal": {
        "description": "Three fronts with populations 500, 150, 30. Agent must rank and commit.",
        "grid_width": 18, "grid_height": 18, "max_turns": 22, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [9, 0]}, {"id": 1, "size": 10, "cell": [9, 17]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3]], "intensity": 38},
            {"cells": [[15, 2], [14, 2], [15, 3]], "intensity": 38},
            {"cells": [[9, 15], [8, 15], [10, 15]], "intensity": 38},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 500, "property_value": 5000, "spreadability": 70},
            {"x": 1,  "y": 2,  "pop": 70,  "property_value": 700,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 70,  "property_value": 700,  "spreadability": 60},
            {"x": 16, "y": 1,  "pop": 150, "property_value": 1500, "spreadability": 65},
            {"x": 16, "y": 2,  "pop": 20,  "property_value": 200,  "spreadability": 60},
            {"x": 15, "y": 1,  "pop": 20,  "property_value": 200,  "spreadability": 60},
            {"x": 9,  "y": 16, "pop": 30,  "property_value": 300,  "spreadability": 55},
            {"x": 8,  "y": 16, "pop": 5,   "property_value": 50,   "spreadability": 50},
            {"x": 10, "y": 16, "pop": 5,   "property_value": 50,   "spreadability": 50},
        ],
    },

    # City siege: central metro threatened from north and south simultaneously.
    "city_siege": {
        "description": "Central city of 700 people attacked from two sides. Crews must split to hold both.",
        "grid_width": 16, "grid_height": 18, "max_turns": 22, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [8, 0]}, {"id": 1, "size": 10, "cell": [8, 17]}],
        "fire_clusters": [
            {"cells": [[6, 4], [7, 4], [8, 4], [7, 5]], "intensity": 35},
            {"cells": [[6, 13], [7, 13], [8, 13], [7, 12]], "intensity": 35},
        ],
        "cells": [
            {"x": 7,  "y": 8,  "pop": 700, "property_value": 7000, "spreadability": 50},
            {"x": 6,  "y": 8,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 8,  "y": 8,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 7,  "y": 7,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 7,  "y": 9,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 7,  "y": 6,  "pop": 80,  "property_value": 800,  "spreadability": 55},
            {"x": 7,  "y": 10, "pop": 80,  "property_value": 800,  "spreadability": 55},
        ],
    },

    # Corridor: three towns along a strip, crews in the middle.
    "corridor": {
        "description": "Three towns along a 20x8 corridor with fires at both ends and the middle.",
        "grid_width": 20, "grid_height": 8, "max_turns": 20, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [10, 4]}, {"id": 1, "size": 10, "cell": [10, 3]}],
        "fire_clusters": [
            {"cells": [[2, 3], [2, 4], [3, 3]], "intensity": 40},
            {"cells": [[10, 1], [11, 1], [10, 0]], "intensity": 30},
            {"cells": [[17, 3], [17, 4], [18, 3]], "intensity": 38},
        ],
        "cells": [
            {"x": 1,  "y": 3,  "pop": 350, "property_value": 3500, "spreadability": 65},
            {"x": 1,  "y": 4,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 0,  "y": 3,  "pop": 50,  "property_value": 500,  "spreadability": 60},
            {"x": 10, "y": 2,  "pop": 200, "property_value": 2000, "spreadability": 55},
            {"x": 11, "y": 2,  "pop": 30,  "property_value": 300,  "spreadability": 50},
            {"x": 9,  "y": 2,  "pop": 30,  "property_value": 300,  "spreadability": 50},
            {"x": 18, "y": 4,  "pop": 280, "property_value": 2800, "spreadability": 65},
            {"x": 19, "y": 4,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 18, "y": 3,  "pop": 40,  "property_value": 400,  "spreadability": 60},
        ],
    },

    # Crescendo: low start intensity but terrain spreads at 85. Must suppress early.
    "crescendo": {
        "description": "Fires start weak but terrain spreadability is 85. Delay costs everything.",
        "grid_width": 14, "grid_height": 14, "max_turns": 18, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [7, 0]}, {"id": 1, "size": 8, "cell": [7, 13]}],
        "fire_clusters": [
            {"cells": [[3, 3], [4, 3]], "intensity": 18},
            {"cells": [[10, 10], [9, 10]], "intensity": 18},
        ],
        "cells": [
            {"x": 2,  "y": 2,  "pop": 280, "property_value": 2800, "spreadability": 85},
            {"x": 1,  "y": 2,  "pop": 40,  "property_value": 400,  "spreadability": 82},
            {"x": 2,  "y": 1,  "pop": 40,  "property_value": 400,  "spreadability": 82},
            {"x": 11, "y": 11, "pop": 240, "property_value": 2400, "spreadability": 85},
            {"x": 12, "y": 11, "pop": 35,  "property_value": 350,  "spreadability": 82},
            {"x": 11, "y": 12, "pop": 35,  "property_value": 350,  "spreadability": 82},
        ],
    },

    # Four villages, two fires: must choose 2 of 4 to save.
    "four_villages": {
        "description": "Four villages, two diagonal fires. Crews can save two; which two?",
        "grid_width": 18, "grid_height": 18, "max_turns": 22, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 10, "cell": [9, 9]}, {"id": 1, "size": 10, "cell": [9, 8]}],
        "fire_clusters": [
            {"cells": [[4, 4], [5, 4], [4, 5]], "intensity": 40},
            {"cells": [[13, 13], [12, 13], [13, 12]], "intensity": 40},
        ],
        "cells": [
            {"x": 2,  "y": 2,  "pop": 220, "property_value": 2200, "spreadability": 65},
            {"x": 2,  "y": 3,  "pop": 30,  "property_value": 300,  "spreadability": 60},
            {"x": 3,  "y": 2,  "pop": 30,  "property_value": 300,  "spreadability": 60},
            {"x": 15, "y": 2,  "pop": 190, "property_value": 1900, "spreadability": 65},
            {"x": 15, "y": 3,  "pop": 25,  "property_value": 250,  "spreadability": 60},
            {"x": 14, "y": 2,  "pop": 25,  "property_value": 250,  "spreadability": 60},
            {"x": 2,  "y": 15, "pop": 170, "property_value": 1700, "spreadability": 65},
            {"x": 2,  "y": 14, "pop": 25,  "property_value": 250,  "spreadability": 60},
            {"x": 3,  "y": 15, "pop": 25,  "property_value": 250,  "spreadability": 60},
            {"x": 15, "y": 15, "pop": 150, "property_value": 1500, "spreadability": 65},
            {"x": 15, "y": 14, "pop": 20,  "property_value": 200,  "spreadability": 60},
            {"x": 14, "y": 15, "pop": 20,  "property_value": 200,  "spreadability": 60},
        ],
    },

    # Three small crews (size 5, 3 AP): forces coordination.
    "small_crews": {
        "description": "Three small crews against three fronts. Each crew alone is marginal.",
        "grid_width": 16, "grid_height": 16, "max_turns": 25, "actions_per_crew": 3,
        "crews": [
            {"id": 0, "size": 5, "cell": [8, 0]},
            {"id": 1, "size": 5, "cell": [8, 15]},
            {"id": 2, "size": 5, "cell": [0, 8]},
        ],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3]], "intensity": 40},
            {"cells": [[13, 2], [12, 2], [13, 3]], "intensity": 40},
            {"cells": [[8, 13], [7, 13], [9, 13]], "intensity": 35},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 300, "property_value": 3000, "spreadability": 65},
            {"x": 1,  "y": 2,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 2,  "y": 1,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 14, "y": 1,  "pop": 280, "property_value": 2800, "spreadability": 65},
            {"x": 14, "y": 2,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 13, "y": 1,  "pop": 40,  "property_value": 400,  "spreadability": 60},
            {"x": 8,  "y": 14, "pop": 200, "property_value": 2000, "spreadability": 60},
            {"x": 7,  "y": 14, "pop": 30,  "property_value": 300,  "spreadability": 55},
            {"x": 9,  "y": 14, "pop": 30,  "property_value": 300,  "spreadability": 55},
        ],
    },

    # Heartland: one enormous city, fires closing from two sides.
    "heartland": {
        "description": "800-person metro at center, two fires converging from the northwest and east.",
        "grid_width": 20, "grid_height": 20, "max_turns": 25, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 12, "cell": [10, 0]}, {"id": 1, "size": 12, "cell": [0, 10]}],
        "fire_clusters": [
            {"cells": [[3, 7], [3, 8], [4, 7]], "intensity": 40},
            {"cells": [[14, 10], [15, 10], [14, 9]], "intensity": 38},
        ],
        "cells": [
            {"x": 9,  "y": 9,  "pop": 800, "property_value": 8000, "spreadability": 45},
            {"x": 8,  "y": 9,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 10, "y": 9,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 9,  "y": 8,  "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 9,  "y": 10, "pop": 100, "property_value": 1000, "spreadability": 50},
            {"x": 6,  "y": 8,  "pop": 150, "property_value": 1500, "spreadability": 55},
            {"x": 12, "y": 9,  "pop": 130, "property_value": 1300, "spreadability": 55},
        ],
    },

    # Last stand: fires already high intensity, short episode, maximum urgency.
    "last_stand": {
        "description": "Fires at intensity 60, adjacent to large population, only 15 turns. No time to travel.",
        "grid_width": 12, "grid_height": 12, "max_turns": 15, "actions_per_crew": 4,
        "crews": [{"id": 0, "size": 12, "cell": [2, 6]}, {"id": 1, "size": 12, "cell": [9, 6]}],
        "fire_clusters": [
            {"cells": [[2, 2], [3, 2], [2, 3], [3, 3]], "intensity": 60},
            {"cells": [[9, 8], [8, 8], [9, 9], [8, 9]], "intensity": 60},
        ],
        "cells": [
            {"x": 1,  "y": 1,  "pop": 450, "property_value": 4500, "spreadability": 70},
            {"x": 1,  "y": 2,  "pop": 60,  "property_value": 600,  "spreadability": 65},
            {"x": 2,  "y": 1,  "pop": 60,  "property_value": 600,  "spreadability": 65},
            {"x": 10, "y": 10, "pop": 400, "property_value": 4000, "spreadability": 70},
            {"x": 10, "y": 9,  "pop": 55,  "property_value": 550,  "spreadability": 65},
            {"x": 9,  "y": 10, "pop": 55,  "property_value": 550,  "spreadability": 65},
        ],
    },
}


def main() -> None:
    seeds = [42, 43, 44]
    results: dict[str, dict] = {}

    for name, scenario in SCENARIOS.items():
        print(f"\n=== {name} ===")
        scenario_results = {}
        for seed in seeds:
            dn = run_raw(scenario, DoNothingAgent(), seed)
            gr = run_raw(scenario, GreedyAgent(), seed)
            spread = gr - dn
            mark = "OK" if spread > 0 else "WARN(greedy<donothing)"
            print(f"  seed={seed}  do_nothing={dn:>12,}  greedy={gr:>12,}  spread={spread:>10,}  {mark}")
            scenario_results[seed] = {"donothing": dn, "greedy": gr}
        results[name] = scenario_results

    print("\n\n=== SEED 42 SUMMARY ===")
    for name, r in results.items():
        dn = r[42]["donothing"]
        gr = r[42]["greedy"]
        print(f"  {name:<22}  donothing={dn:>12,}  greedy={gr:>12,}  spread={gr-dn:>10,}")


if __name__ == "__main__":
    main()
