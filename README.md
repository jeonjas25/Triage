# Triage — Wildfire Resource Allocation Benchmark

SWECCATHON 2026 &nbsp;|&nbsp; [mesocosm](https://mesocosm.swecc.org) benchmark

## What is Triage?

Triage is a turn-based wildfire emergency management benchmark for the mesocosm platform. An AI agent commands fire crews across a burning grid, deciding each turn where to move, which fires to suppress, which terrain to prepare, and when to evacuate civilians.

The core tension: fire spreads exponentially, crews are scarce, and optimal resource allocation under time pressure determines how many lives are saved.

---

## Scenarios

15 built-in scenarios spanning a range of tactical challenges:

| Scenario | Grid | Turns | Crews | Challenge |
|---|---|---|---|---|
| `two_towns` | 14×14 | 20 | 2 × size 10 | Symmetric opening |
| `three_fronts` | 18×18 | 25 | 2 × size 10 | Three simultaneous fronts |
| `squeeze` | 16×16 | 22 | 2 × size 7 | Pinched terrain, limited AP |
| `asymmetric_stakes` | 14×14 | 20 | 2 × size 10 | Unequal population stakes |
| `racing_fronts` | 16×12 | 18 | 2 × size 9 | Fast-moving converging fires |
| `delayed_storm` | 18×18 | 25 | 2 × size 10 | Hidden third fire ignites at turn 6 |
| `one_crew` | 12×12 | 20 | 1 × size 15 | Single large crew, all fronts |
| `three_unequal` | 18×18 | 22 | 2 × size 10 | Skewed population distribution |
| `city_siege` | 16×18 | 22 | 2 × size 10 | Dense urban center under threat |
| `corridor` | 20×8 | 20 | 2 × size 10 | Linear corridor, fires at both ends |
| `crescendo` | 14×14 | 18 | 2 × size 10 | Weak fires, 85% spreadability — delay costs everything |
| `four_villages` | 18×18 | 22 | 2 × size 10 | Four villages; crews can only save two |
| `small_crews` | 16×16 | 25 | 3 × size 5 | Three marginal crews against three fronts |
| `heartland` | 20×20 | 25 | 2 × size 12 | 800-person metro, two converging fires |
| `last_stand` | 12×12 | 15 | 2 × size 12 | Intensity-60 fires, 15 turns, no margin |

---

## Scoring

**Per-turn delta:**

| Event | Points |
|---|---|
| Safe person alive | +10 |
| Person on burning cell (each tick) | −10 |
| Non-burning cell | +1 |
| Burning cell | −5 |
| Casualty (death) | −100 |
| Newly ignited cell | −1,000 |
| Property destroyed | −1 per unit |

**Terminal normalization:**

```
norm_score = (raw_score − baseline_do_nothing) / (baseline_greedy − baseline_do_nothing)
```

- `0.0` = matches the do-nothing baseline  
- `1.0` = matches the greedy heuristic  
- `>1.0` = beats greedy (the goal)

---

## Actions

Each turn the agent submits a list of commands — up to `actions_per_crew` per crew:

```json
{
  "commands": [
    {"crew": 0, "action": "move",     "to": [3, 4]},
    {"crew": 0, "action": "suppress"},
    {"crew": 1, "action": "prep"},
    {"crew": 1, "action": "evacuate", "to": [7, 3]}
  ]
}
```

| Command | AP cost | Effect |
|---|---|---|
| `move` | 1 | Move crew to adjacent cell `[x, y]` via `"to"` |
| `suppress` | 1 | Reduce intensity of crew's current cell by `crew_size × 3`; extinguishes at 0 |
| `prep` | 1 | Reduce spreadability of crew's current cell by `crew_size × 3` |
| `evacuate` | 1 | Carry up to `crew_size × 5` people from current cell to adjacent cell via `"to"` |

---

## Observation

The agent receives a **partial observation** — only cells that are on fire, adjacent to fire, have population, or have a crew:

```json
{
  "turn": 5,
  "max_turns": 20,
  "grid_width": 14,
  "grid_height": 14,
  "score": -12340,
  "crews": [
    {"id": 0, "cell": [3, 4], "size": 10, "actions_remaining": 4}
  ],
  "cells": [
    {
      "x": 3, "y": 3,
      "pop": 50, "on_fire": true, "intensity": 62,
      "spreadability": 65, "property_value": 500, "property_remaining": 320
    }
  ]
}
```

---

## Running Locally

```bash
pip install -r requirements.txt

# Run greedy agent on the heartland scenario
python run_local.py --agent greedy --scenario heartland

# Run with a specific seed
python run_local.py --agent greedy --scenario two_towns --seed 42

# Save a replay for visualization
python run_local.py --agent greedy --scenario heartland \
    --trace-out showcase/data/replay.json

# Recompute baselines for all 15 scenarios
python compute_baselines.py
```

---

## Replay Visualization

```bash
# 1. Generate a local trace
python run_local.py --agent greedy --scenario heartland \
    --trace-out showcase/data/replay.json

# 2. Serve showcase/ and open in browser
cd showcase
python -m http.server 8080
# Visit: http://localhost:8080
```

The visualizer shows the grid animating turn-by-turn, crew positions, fire spread with intensity glow, and per-step stats. Use ⏮⏪▶⏩ to navigate or click the progress bar to scrub.

You can also export a real mesocosm run:
```bash
mesocosm run export RUN_ID -o showcase/data/replay.json
```

---

## Project Structure

```
auxiliary/            mesocosm submission package
  adapter.py          HTTP adapter (entry point for platform)
  env.py              TriageEnv — core environment class
  env_core/
    fire.py           Cellular automaton fire spread model
    crews.py          Crew actions: move / suppress / prep / evacuate
    grid.py           Grid + Cell dataclasses, scenario loader
    casualty.py       Per-tick scoring formula
  benchanything.json  mesocosm manifest

baselines/
  agents.py           DoNothing / Random / Greedy reference agents

scenarios/            15 scenario JSON files (with baked baselines)

showcase/
  index.html          Web replay visualizer
  data/               Place replay.json here (gitignored)

run_local.py          Local debug runner + trace exporter
compute_baselines.py  Regenerate scenario baselines
```
