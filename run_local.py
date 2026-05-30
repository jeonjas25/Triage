"""
Debug loop: run a local agent through the Triage env and print results.

Usage:
    python run_local.py                    # RandomAgent, default scenario, seed 42
    python run_local.py --agent greedy
    python run_local.py --seed 7 --turns 30
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Allow importing from auxiliary/ (env_core lives there)
sys.path.insert(0, str(Path(__file__).parent / "auxiliary"))

from env import TriageEnv  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from baselines.agents import DoNothingAgent, GreedyAgent, RandomAgent  # noqa: E402


def run_episode(agent, seed: int, scenario: str | None) -> dict:
    env = TriageEnv()
    kwargs: dict = {}
    if scenario:
        kwargs["scenario"] = scenario
    obs = env.reset(seed=seed, **kwargs)
    step = 0

    while True:
        action = agent.act(obs)
        result = env.step(action)
        obs = result.observation
        step += 1
        if result.terminated or result.truncated:
            print(f"  Episode ended at turn {step} (terminated={result.terminated})")
            return result.info


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a local Triage episode.")
    parser.add_argument("--agent", choices=["random", "greedy", "nothing"], default="random")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=1)
    parser.add_argument("--scenario", default=None, help="Scenario name (e.g. two_towns, three_fronts, squeeze)")
    args = parser.parse_args()

    agents = {"random": lambda s: RandomAgent(seed=s), "greedy": lambda s: GreedyAgent(), "nothing": lambda s: DoNothingAgent()}

    for ep in range(args.episodes):
        seed = args.seed + ep
        label = args.scenario or "default"
        print(f"\n=== Episode {ep + 1} | scenario={label} seed={seed} agent={args.agent} ===")
        agent = agents[args.agent](seed)
        info = run_episode(agent, seed, args.scenario)
        print(json.dumps(info, indent=2))


if __name__ == "__main__":
    main()
