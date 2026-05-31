"""
Recompute do-nothing and greedy baseline running totals for all scenario JSON files.
Writes baseline_donothing and baseline_greedy back into each scenario file.

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

SCENARIOS_DIR = Path(__file__).parent / "scenarios"
SEEDS = [42, 43, 44]


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


def main() -> None:
    scenario_files = sorted(SCENARIOS_DIR.glob("*.json"))
    if not scenario_files:
        print("No scenario JSON files found in", SCENARIOS_DIR)
        return

    results: dict[str, dict] = {}

    for path in scenario_files:
        name = path.stem
        with open(path) as f:
            scenario = json.load(f)

        print(f"\n=== {name} ===")
        seed_results: dict[int, dict] = {}
        for seed in SEEDS:
            dn = run_raw(scenario, DoNothingAgent(), seed)
            gr = run_raw(scenario, GreedyAgent(), seed)
            spread = gr - dn
            mark = "OK" if spread > 0 else "WARN(greedy<=donothing)"
            print(f"  seed={seed}  do_nothing={dn:>12,}  greedy={gr:>12,}  spread={spread:>10,}  {mark}")
            seed_results[seed] = {"donothing": dn, "greedy": gr}
        results[name] = seed_results

        # Bake seed-42 baselines into the JSON file
        scenario["baseline_donothing"] = seed_results[42]["donothing"]
        scenario["baseline_greedy"]    = seed_results[42]["greedy"]
        with open(path, "w") as f:
            json.dump(scenario, f, indent=2)
        print(f"  → wrote baselines to {path.name}")

    print("\n\n=== SEED 42 SUMMARY ===")
    for name, r in results.items():
        dn = r[42]["donothing"]
        gr = r[42]["greedy"]
        spread = gr - dn
        print(f"  {name:<22}  donothing={dn:>12,}  greedy={gr:>12,}  spread={spread:>10,}")


if __name__ == "__main__":
    main()
