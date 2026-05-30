# Firebreak: a wildfire resource-allocation benchmark

**SWECCATHON 2026 build plan.** Track: AGI & real-world modeling. Single hazard (wildfire), multiple simultaneous fire patches competing for one shared budget. Firebreak is the working name; flag it if you want to change it.

This is the shared source of truth. If you are picking up a workstream, read sections 1 through 5 first so your part matches everyone else's. Each phase ends with a "Done when" line. Do not start a later phase's integration until the earlier phase hits its Done-when.

---

## 0. How to use this doc with Claude

Each of us is working with Claude. To stay in sync, paste the relevant section into your Claude when starting a task, and tell it which file you own (section 4). The schemas in section 5 are frozen contracts: if your Claude wants to change an observation or action field, flag it in the team channel first, because everyone's code depends on those shapes.

Writing-style convention for all written deliverables (README, manifest descriptions, video script): lean natural prose, no em dashes.

---

## 1. What we are building, in one paragraph

A grid-based simulation where an LLM agent fights an unfolding wildfire under a fixed resource budget. Several separate fire patches ignite across the same map, each with its own fuel, wind exposure, and population at risk, and they all draw from one shared budget. The agent sees the current map and a noisy forecast, and each turn it decides how to spend limited resources to save the most lives. The benchmark measures how well a model allocates scarce resources across competing fire fronts. That competition, "the northern fire and the southern fire both want my crews and I can only afford to hold one line," is the entire point.

---

## 2. How the hackathon scores us (design to this)

Judging weights:

- **Track execution and wow, 40%.** The wow shot is an animated map with several fires spreading at once while a shrinking budget forces the agent to let one front burn.
- **Usefulness and gap, 30%.** Test cases and grading logic must live in our code and config. This is why scoring is declarative (section 6) and why we ship diagnostic metrics.
- **Data and reusability, 20%.** Our seeded scenarios plus recorded traces are the reusable artifact. Ship them in the repo.
- **Presentation clarity, 10%.** README plus a 1 to 2 minute demo video with voiceover.

Submission (Devpost, by Mon June 1, 4:00 PM): GitHub repo, a UI that visualizes traces, and the demo video as an unlisted YouTube link.

---

## 3. How mesocosm works (the platform that runs us)

Mesocosm is the harness and it brings its own agents. We do **not** write the agent, the prompt, or the action parser. We ship an environment; mesocosm runs models against it and records traces.

The contract:

- We implement a gym-style env: `reset(seed)` and `step(action)` returning `StepResult(observation, reward, terminated, truncated, info)`.
- `adapter.py` wraps the env in an HTTP server exposing `/health`, `/reset`, `/step`, `/close` (optional `/render`). Scaffold this with `bench init`, do not hand-write it.
- `benchanything.json` is the manifest: adapter file, name, description, a `binding_vow` (observation space, action space, reward, episode config), and a `scoring` block.
- **The environment never grades itself.** `step` emits a reward plus an `info` dict of raw signals. The `scoring` block decides the verdict by aggregating those signals. Keep verdicts out of `env.py`.
- Resets must be deterministic and seed-stable: same seed, same scenario, every time.

Flow: write env, `pip install swecc-mesocosm`, test the four endpoints locally with curl, submit the repo URL through the form or API, watch status go pending, cloning, ready, then run Test Bench or Full Bench.

---

## 4. Repo structure and ownership

```
firebreak-bench/
  benchanything.json     # manifest + scoring          [owner: Integration]
  adapter.py             # boilerplate from bench init  [owner: Integration]
  env.py                 # BaseEnv subclass             [owner: Engine]
  requirements.txt
  env_core/
    grid.py              # grid + scenario generator    [owner: Engine]
    casualty.py          # casualty model               [owner: Engine]
    resources.py         # resource costs + effects     [owner: Engine]
    fire.py              # wildfire cellular automaton   [owner: Fire model]
  scenarios/             # seeded configs + baselines   [owner: Scenarios]
  baselines/
    agents.py            # random + greedy (local only) [owner: Scenarios]
  run_local.py           # debug loop + trace dumps     [owner: Integration]
  viz/                   # built last, from traces      [owner: Viz]
  README.md
```

Workstreams: **Engine, Fire model, Scenarios, Integration, Viz.** Assign one owner each. If we are short on people, one person can hold Engine plus Fire model, and Integration plus Viz pair naturally. If solo, just follow the phase order top to bottom.

---

## 5. Frozen contracts (do not change without a team heads-up)

### Observation (what the agent sees each turn)

```json
{
  "turn": 3,
  "max_turns": 20,
  "budget_remaining": 45,
  "grid_width": 12,
  "grid_height": 12,
  "cells": [
    {"x": 4, "y": 5, "pop": 120, "evacuated": 0,
     "fire": 0.3, "fuel": 0.8,
     "resources": ["crew"]}
  ],
  "resources_available": {"crew": 2, "barrier": 5, "shelter_capacity": 300},
  "forecast": {"wind_dir": "NE", "wind_strength": "rising", "reliability": "noisy"}
}
```

Critical: the forecast is intentionally noisy and never exposes the true wind schedule or the ignition probabilities. Fuel is visible because it is static terrain a firefighter would know, but the spread rule itself stays hidden. If the agent can see the exact spread math, it solves the physics instead of planning under uncertainty, which kills the skill we are measuring.

### Action (one action equals one full turn of allocation)

`action_space.type` is `"json"`. The agent returns its whole turn at once, then the world ticks once.

```json
{
  "allocations": [
    {"type": "evacuate", "cell": [4, 5]},
    {"type": "deploy_crew", "cell": [6, 2], "crew_id": 1},
    {"type": "build_barrier", "cell": [3, 3]},
    {"type": "open_shelter", "cell": [0, 0]}
  ]
}
```

A barrier is a firebreak: placed on a cell, it blocks fire from spreading through it. Validation rule: actions that exceed the remaining budget or are otherwise illegal are rejected and counted (`illegal_actions`), never crashed on.

### Resource model

- **Crews are reusable.** They are repositioned each turn (a crew can drive from the northern front to the southern one). This is what makes allocation rich and job-like.
- **Barriers (firebreaks) are consumable.** Once placed, the spend is gone.
- All costs are integers. No continuous spend, ever, or the verifier gets mushy.

### Casualty model (keep it simple and deterministic)

Per cell, per turn:

```
threat     = fire intensity on the cell, clamped to [0, 1]
mitigation = reduction from crews present on the cell
casualties = round(unevacuated_pop * threat * (1 - mitigation))
```

Evacuation lowers `unevacuated_pop` over a turn or two (it is not instant). Multiple fires need no special handling here: every burning cell is scored the same way regardless of which patch it belongs to.

### Fire model (the spread engine)

```python
class FireModel:
    def step(self, grid, schedule_t): ...    # advance every burning cell one tick
    def intensity(self, cell) -> float: ...   # fire intensity on this cell now, in [0, 1]
```

One model, many patches. Multiple simultaneous fires are just multiple sets of burning cells on the same grid, seeded from different ignition points. There is no per-fire bookkeeping: a "patch" is an emergent cluster, not a separate object. A firebreak cell simply refuses to ignite.

### Determinism rule (mesocosm enforces this)

Pre-roll all world randomness into a fixed schedule at `reset`, indexed by turn (wind per turn, ignition noise per cell per turn). The fire model reads `schedule[turn]`, it never draws from a live RNG inside `step`. Reason: two agents who act differently must still face identical weather, so outcome differences are skill, not luck. A live RNG desyncs the moment action counts differ.

---

## 6. Scoring (lives in benchanything.json, not in code)

The env emits a normalized score as a terminal signal. Normalization happens in the env because only the env knows each scenario's do-nothing floor and greedy ceiling, and those two numbers are baked into the scenario config (Phase 3), so `reset` just reads them.

`step` on the terminal turn:

```python
lives_saved = self._pop_initial - self._casualties
denom = max(1, self._baseline_greedy - self._baseline_donothing)
norm  = (lives_saved - self._baseline_donothing) / denom
info = {
    "norm_score": round(norm, 4),
    "lives_saved": lives_saved,
    "lives_lost": self._casualties,
    "budget_used": self._spent,
    "illegal_actions": self._illegal,
}
```

`benchanything.json` scoring block:

```json
"scoring": {
  "primary_metric": "rescue_score",
  "higher_is_better": true,
  "metrics": [
    {"name": "rescue_score", "type": "terminal_field", "field": "norm_score", "aggregation": "mean"},
    {"name": "lives_lost",  "type": "terminal_field", "field": "lives_lost", "aggregation": "mean"},
    {"name": "budget_used", "type": "terminal_field", "field": "budget_used", "aggregation": "mean"},
    {"name": "illegal_rate","type": "terminal_field", "field": "illegal_actions", "aggregation": "mean"}
  ]
}
```

The three diagnostics are not decoration. They catch the obvious exploit (an agent that scores well only by blowing the budget or spamming illegal moves). If `budget_used` or `illegal_rate` looks wrong, the headline score is hollow.

---

## 7. Build phases

Guiding principle: get a random agent playing a full episode and printing a score **before** anything is good. That surfaces integration bugs while we still have time. The MVP is the safety net.

### Phase 0 — Setup and freeze the contracts (Saturday, first 2 to 3 hours)

Owner: everyone, together.

1. Create the public GitHub repo, add everyone as collaborators.
2. `pip install swecc-mesocosm`, run `bench init` to scaffold `adapter.py` and a starter `benchanything.json`. Note the real `BaseEnv` and `StepResult` signatures from the SDK and match them exactly.
3. Lock section 5 schemas into the repo as `CONTRACTS.md` so nobody drifts.
4. Stub the file tree from section 4 with empty classes that import cleanly.

**Done when:** the repo imports without errors and `adapter.py` starts and answers `/health` with the empty stub env.

### Phase 1 — Engine plus one fire, end to end (Saturday, rest of day)

Owners: Engine, Fire model, Scenarios in parallel.

1. Engine: `grid.py` (grid plus scenario generator), `casualty.py`, `resources.py`, and the `env.py` `reset`/`step` loop. Wire the pre-rolled schedule.
2. Fire model: `fire.py` as a cellular automaton (ignite-by-neighbor weighted by wind and fuel, a cell burns k turns then dies). Seed multiple ignition points so several patches grow at once from the start.
3. Scenarios: `baselines/agents.py` with a `RandomAgent` and a `GreedyAgent` (greedy = always send resources to the highest threat-times-population cell affordable). Write `run_local.py` to play an agent through one scenario and print the raw `info`.
4. Compute and hardcode the do-nothing and greedy baselines for 2 or 3 test scenarios so normalization returns sane numbers.

**Done when:** `python run_local.py` runs a RandomAgent through a multi-patch fire scenario start to finish and prints a `norm_score`. This is the MVP.

### Phase 2 — Mesocosm integration (Saturday night or Sunday morning)

Owner: Integration.

1. Fill in `benchanything.json`: manifest fields, `binding_vow` (observation json, action json with the section 5 format described in plain language so the agent knows the shape, reward range, `max_steps`, `deterministic_reset: true`, `supports_seed: true`), and the section 6 scoring block.
2. Run the slide-7 local checklist: start `adapter.py` on a port, curl `/health`, `/reset` (confirm same seed gives same first observation), `/step`, `/close`.
3. Submit the repo URL through the form or API. Confirm status reaches ready.
4. Run a Test Bench against a **cheap or local model first** (DeepSeek, Ollama). The deck says to do this before spending mesocosm or Cursor credits. Pull one real trace.

**Done when:** a real model has played at least one full episode through mesocosm and we have a trace file in hand.

### Phase 3 — Multi-patch difficulty and the budget squeeze (Sunday)

Owners: Fire model, Scenarios, Engine support.

1. Fire model plus Scenarios: make the patches differ. Vary per-patch ignition timing, fuel, wind exposure, and the population behind each front, so the fires are not interchangeable and the agent has to rank them.
2. Scenarios: build the final fixed set of 10 to 20 seeded scenarios. Include several where two or three patches threaten different population centers and the budget cannot hold every line.
3. Engine plus Scenarios: tune the budget to roughly 50 to 70 percent of full-coverage cost. Too much budget makes "do everything" optimal and tests nothing. The squeeze is the benchmark.
4. Recompute do-nothing and greedy baselines for every final scenario and bake them into the scenario configs.

**Done when:** there is a scenario where a model visibly has to let one patch burn to save a more valuable one, and the scenario set with baked baselines is committed.

### Phase 4 — Visualization, video, and submission (Sunday night into Monday morning)

Owners: Viz lead, everyone on polish.

1. Viz: point an LLM at the trace JSON and have it build a web animation of the grid over time, fires spreading, crews and firebreaks deploying, a running casualty and budget counter. The deck is explicit that the viz is the easy part once traces exist.
2. Record the 1 to 2 minute video: open on several fires growing at once, show the agent's allocation choices and the front it chooses to abandon, end on the score versus the greedy baseline. Voiceover, no face needed. Upload as unlisted YouTube.
3. Write the README: what Firebreak measures, how to run it, the scenario set, the scoring rationale, and how to reuse the traces.
4. Submit on Devpost: team, repo, demo site, video.

**Done when:** all four Devpost items are attached. Target this for Monday late morning. Leave the afternoon before 4 PM as pure buffer.

---

## 8. Timeline at a glance

- **Sat AM:** Phase 0.
- **Sat PM:** Phase 1 (MVP done by end of day).
- **Sat night / Sun AM:** Phase 2 (real trace in hand).
- **Sun:** Phase 3 (multi-patch difficulty, scenarios, budget tuned).
- **Sun night / Mon AM:** Phase 4 (viz, video, README, submit).
- **Mon, by 4:00 PM:** submitted, with buffer to spare.

---

## 9. Risks and fallbacks

- **Simulator slips.** The whole project rests on the engine. If Phase 1 runs long, ship with fewer patches and a smaller grid. A clean two-patch env beats a broken five-patch one. Difficulty is a dial we turn down at the end, not a thing we cut.
- **Budget credits run dry.** Develop and debug entirely against local or cheap models. Save expensive mesocosm runs for the final scoring pass only.
- **Determinism bugs.** If two identical-seed runs diverge, the pre-rolled schedule is being bypassed somewhere. This is the most likely silent bug. Test it explicitly in Phase 2 step 2.
- **Reward hacking in the demo.** Watch the diagnostics. A high score with junk `budget_used` or `illegal_rate` is not a real result, and a judge may ask.

---

## 10. Two traps we keep reminding each other about

1. **Never expose the spread rule to the agent.** Forecast stays noisy, true wind and ignition probabilities stay hidden. The skill is planning under uncertainty, not reading the physics off the screen.
2. **Action space stays discrete with integer costs.** The moment allocation goes continuous, the verifier gets mushy and scope explodes.