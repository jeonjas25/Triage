"""
Compute do-nothing and greedy baseline running totals for all scenarios and
bake them back into the scenario JSON files.

Run this whenever scenarios change.

Usage:
    python compute_baselines.py            # print baselines for every scenario
    python compute_baselines.py --write    # also write them into scenarios/*.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "auxiliary"))
sys.path.insert(0, str(ROOT))

from env import TriageEnv
from baselines.agents import DoNothingAgent, GreedyAgent

SCENARIOS_DIR = ROOT / "scenarios"
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
    write = "--write" in sys.argv
    for path in sorted(SCENARIOS_DIR.glob("*.json")):
        scenario = json.loads(path.read_text())
        name = scenario.get("name", path.stem)
        print(f"\n=== {name} ===")
        seed42 = None
        for seed in SEEDS:
            dn = run_raw(scenario, DoNothingAgent(), seed)
            gr = run_raw(scenario, GreedyAgent(), seed)
            spread = gr - dn
            mark = "OK" if spread > 0 else "WARN(greedy<=donothing)"
            print(f"  seed={seed}  do_nothing={dn:>12,}  greedy={gr:>12,}  spread={spread:>10,}  {mark}")
            if seed == 42:
                seed42 = (dn, gr)

        if write and seed42:
            scenario["baseline_donothing"], scenario["baseline_greedy"] = seed42
            path.write_text(json.dumps(scenario, indent=2) + "\n")
            print(f"  wrote baselines (seed 42) -> {path.name}")


if __name__ == "__main__":
    main()
