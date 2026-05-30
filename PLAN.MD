# Triage: a wildfire crew-allocation benchmark

**SWECCATHON 2026 build plan.** Track: AGI and real-world modeling. Single hazard
(wildfire), multiple simultaneous fire fronts competing for one small pool of
firefighting crews. Triage is the working name.

This is the shared source of truth. If you are picking up a workstream, read
sections 1 through 6 first so your part matches everyone else's. Each phase ends
with a "Done when" line. Do not start a later phase's integration until the
earlier phase hits its Done-when.

---

## 0. How to use this doc with Claude

Each of us is working with Claude. To stay in sync, paste the relevant section
into your Claude when starting a task, and tell it which file you own (section 4).
The mechanics in sections 5 and 6 are fixed: if your Claude wants to change an
action, an observation field, or the spread or scoring rule, flag it in the team
channel first, because everyone's code depends on those shapes. Scenario
parameters (board size, crews, fires, population) are free to vary and do not need
a heads-up.

Writing-style convention for all written deliverables (README, manifest
descriptions, video script): lean natural prose, no em dashes.

---

## 1. What we are building, in one paragraph

A grid-based simulation where an LLM agent fights an unfolding wildfire by
directing a small set of firefighting crews. Several separate fire fronts ignite
across one map and grow at once. Each turn the agent commands every crew, spending
a fixed pool of action points per crew on moving, suppressing fire, prepping
firebreaks, and evacuating people. Crews are few and action points are scarce, so
the northern front and the southern front both want crews the agent cannot afford
to send to both. The benchmark measures how well a model triages scarce responders
across competing fire fronts under that pressure. That competition, "both fronts
want my crews and I can only hold one line," is the entire point.

---

## 2. How the hackathon scores us (design to this)

Judging weights:

- **Track execution and wow, 40%.** The wow shot is an animated map with several
  fires spreading at once while too few crews force the agent to let one front
  burn and concentrate on saving people elsewhere.
- **Usefulness and gap, 30%.** Test cases and grading logic must live in our code
  and config. This is why scoring is declarative (section 6) and why we ship
  diagnostic metrics.
- **Data and reusability, 20%.** Our seeded scenarios plus recorded traces are the
  reusable artifact. Ship them in the repo.
- **Presentation clarity, 10%.** README plus a 1 to 2 minute demo video with
  voiceover.

Submission (Devpost, by Mon June 1, 4:00 PM): GitHub repo, a UI that visualizes
traces, and the demo video as an unlisted YouTube link.

---

## 3. How mesocosm works (the platform that runs us)

Mesocosm is the harness and it brings its own agents. We do **not** write the
agent, the prompt, or the action parser. We ship an environment; mesocosm runs
models against it and records traces.

The contract:

- We implement a gym-style env: `reset(seed)` and `step(action)` returning
  `StepResult(observation, reward, terminated, truncated, info)`.
- `adapter.py` wraps the env in an HTTP server exposing `/health`, `/reset`,
  `/step`, `/close` (optional `/render`). Scaffold this with `bench init`, do not
  hand-write it.
- `benchanything.json` is the manifest: adapter file, name, description, a
  `binding_vow` (observation space, action space, reward, episode config), and a
  `scoring` block.
- **The environment never grades itself.** `step` emits a reward plus an `info`
  dict of raw signals. The `scoring` block decides the verdict by aggregating
  those signals. Keep verdicts out of `env.py`.
- Resets must be deterministic and seed-stable: same seed, same scenario, every
  time.

Flow: write env, `pip install swecc-mesocosm`, test the four endpoints locally
with curl, submit the repo URL through the form or API, watch status go pending,
cloning, ready, then run Test Bench or Full Bench.

---

## 4. Repo structure and ownership

```
triage-bench/
  benchanything.json     # manifest + scoring          [owner: Integration]
  adapter.py             # boilerplate from bench init  [owner: Integration]
  env.py                 # BaseEnv subclass             [owner: Engine]
  requirements.txt
  env_core/
    grid.py              # grid + scenario generator    [owner: Engine]
    casualty.py          # casualty + property model    [owner: Engine]
    crews.py             # crew state + action effects   [owner: Engine]
    fire.py              # wildfire cellular automaton   [owner: Fire model]
  scenarios/             # seeded configs + baselines   [owner: Scenarios]
  baselines/
    agents.py            # random + greedy (local only) [owner: Scenarios]
  run_local.py           # debug loop + trace dumps     [owner: Integration]
  viz/                   # built last, from traces      [owner: Viz]
  README.md
```

Workstreams: **Engine, Fire model, Scenarios, Integration, Viz.** Assign one owner
each. If we are short on people, one person can hold Engine plus Fire model, and
Integration plus Viz pair naturally. If solo, just follow the phase order top to
bottom.

---

## 5. The simulation

Three things are fixed across every scenario: the four actions, the fire-spread
and burnout rule, and the scoring. Everything else, the board, the crews, the
fires, the people, is a scenario dial.

### What is fixed (identical in every scenario)

**Crews and their turn.** Every crew receives the same pool of action points at
the start of each turn, spendable across move, suppress, prep, and evacuate in any
mix. Crews have identity and position, both exposed to the agent. Crews may share
a cell, may stack on a cell, and may work the same cell together with their
effects summing. Crew size scales effect, never cost. The one exception is
movement, which costs the same per cell for every crew regardless of size. There
is no separate "bigger crews need fewer actions" rule; that outcome falls out of
effect-scaling on its own.

The four actions:

- **move**: relocate the crew to an adjacent cell. Costs 1 action per cell, so
  traveling N cells is N move commands. Size-independent.
- **suppress**: lower the intensity of the cell the crew is on by `crew_size * 3`
  points per action, floor 0. Stacked crews sum their reductions. A cell driven to
  intensity 0 by crews is extinguished.
- **prep**: lower the spreadability of the cell the crew is on by `crew_size * 3`
  points per action, floor 0. The trench-digging move: it makes the cell harder to
  ignite from neighbors.
- **evacuate**: move the crew to an adjacent cell and carry up to `crew_size * 5`
  people from the current cell along with it, 1 action. The crew moves with the
  people, so an evacuate is also a move. The agent decides how far to ferry
  people; there is no requirement to reach safety in one step. People count as
  safe at scoring time when they sit on a non-burning cell.

All action effects are integers. Nothing about the action space is continuous.

**The fire-spread and burnout rule.** Resolved during the world tick, after the
crews' commands are applied. There is no wind. Spread is isotropic.

- Each tick, every burning cell may grow in intensity by a pre-rolled increment
  per turn, capped at 100.
- Each burning cell may ignite its non-burning neighbors. Ignition likelihood
  rises with the neighbor's spreadability and the burning cell's intensity.
  Whether a neighbor actually ignites is decided by comparing a pre-rolled roll
  for that cell and turn against the computed likelihood. The agent sees
  spreadability, which is the risk, but not the roll, so it cannot know in advance
  which at-risk cells will catch.
- Each cell, when it ignites, is assigned a hidden burn timer pre-rolled in the
  range 12 to 20 turns. The timer counts down each turn the cell stays on fire; at
  zero the cell burns out to intensity 0 and cannot reignite. Burnout is a slow
  backstop and the migrating-front look, not the main way fires end. Inside a
  normal episode the brunt of fire clearing is the crews.

**The casualty and property rule.**

- Casualties: per burning cell, casualties scale with intensity and the
  unevacuated population on that cell. Higher intensity kills a larger share.
  Population moved off a burning cell, or whose cell was driven to intensity 0 this
  turn, is not at risk this tick.
- Property: each cell tracks the peak intensity it has ever reached. Loss is
  locked to that peak. A cell worth P that has peaked at intensity I has lost
  `P * I / 100`. A tick that pushes a cell to a new peak adds
  `P * (new_peak - old_peak) / 100` of loss, which feeds the property term in
  scoring.

**Determinism.** Pre-roll all world randomness into a fixed schedule at `reset`,
indexed by turn: the ignition roll per cell per turn, the intensity-growth
increment per cell per turn, and each cell's burn timer. The tick reads
`schedule[turn]` and never draws from a live RNG inside `step`. Two agents who act
differently must still face identical conditions, so outcome differences are
skill, not luck. A live RNG desyncs the moment action counts differ.

### What each scenario sets

A scenario is free to vary all of the following. This is where difficulty lives.

- grid width and height
- number of crews, each crew's size, each crew's start cell (crews may share one)
- number, placement, and starting intensity of the initial fires, seeded as
  clusters of adjacent cells so each fire reads as a front, not a dot
- per-cell population
- per-cell original property value
- per-cell initial spreadability
- action points per crew per turn
- max turns
- the baked do-nothing and greedy baselines for normalization

### Observation (what the agent sees each turn)

On a large board, the observation sends only decision-relevant cells: any cell on
fire, adjacent to fire, holding population, or holding a crew. Empty cells far from
fire carry no decision value and are omitted. The agent always gets full grid
dimensions.

```json
{
  "turn": 3,
  "max_turns": 20,
  "grid_width": 24,
  "grid_height": 24,
  "score": 128400,
  "crews": [
    {"id": 0, "size": 20, "cell": [4, 5], "actions_remaining": 4},
    {"id": 1, "size": 12, "cell": [9, 9], "actions_remaining": 4}
  ],
  "cells": [
    {"x": 4, "y": 5, "pop": 300, "on_fire": true,  "intensity": 40,
     "spreadability": 0,  "property_value": 3000, "property_remaining": 1800},
    {"x": 5, "y": 5, "pop": 250, "on_fire": false, "intensity": 0,
     "spreadability": 70, "property_value": 2600, "property_remaining": 2600}
  ]
}
```

### Action (one submission equals one full turn for all crews)

`action_space.type` is `"json"`. The agent returns one list of commands covering
every crew, then the world ticks once. Commands are processed per crew in listed
order, each consuming 1 action point, until that crew's pool is exhausted. A
command that would exceed a crew's remaining actions, or is otherwise illegal
(non-adjacent move, suppressing a cell the crew is not on, an unknown crew), is
rejected and counted in `illegal_actions`, never crashed on.

```json
{
  "commands": [
    {"crew": 0, "action": "move",     "to": [5, 5]},
    {"crew": 0, "action": "suppress"},
    {"crew": 1, "action": "evacuate", "to": [8, 9]},
    {"crew": 1, "action": "prep"}
  ]
}
```

---

## 6. Scoring (fixed, lives in benchanything.json, not in code)

### Per-tick raw score

After each tick resolves, the env adds a signed delta to a running total:

```
+10  per person on a non-burning cell
-10  per person on a burning cell
+1   per non-burning cell            (minor)
-5   per burning cell                (minor)
-100 per casualty this tick
-1000 per cell newly ignited this tick
-1   per point of property value lost this tick
```

Casualties are the dominant cost. The -1000 ignition penalty is a flat, one-time
charge that discourages letting fire creep into empty terrain, but it is small
next to what happens when a populated cell catches: a few hundred people at -100
each runs into the tens of thousands, plus the ongoing -10 per person per turn
while they burn. So wherever people are, the casualty term swamps everything and
the ignition penalty is close to a rounding error. The objective the weights
encode is protect populated cells first, then contain spread into empty land, then
property. The per-cell bare terms are cosmetic.

### Terminal normalization

Rewarding every safe person every turn means most of the raw total is just the
static safe area, so the marginal signal would drown. The env normalizes the
running total against the scenario's do-nothing floor and greedy ceiling, both
computed on the identical pre-rolled schedule the agent faces and baked into the
scenario config.

`step` on the terminal turn:

```python
denom = max(1, self._baseline_greedy - self._baseline_donothing)
norm  = (self._running_total - self._baseline_donothing) / denom
info = {
    "norm_score": round(norm, 4),
    "lives_lost": self._casualties,
    "property_lost": self._property_lost,
    "cells_burned": self._cells_ignited,
    "illegal_actions": self._illegal,
}
```

`norm` can fall below 0 (worse than nothing) or above 1 (better than greedy). We
do not clamp, because beating greedy is exactly the leaderboard signal we want.
Set the reward range wide enough to allow it, for example low -1.0, high 2.0, with
a note that the practical band is 0 to 1.

The episode terminates when no cells are on fire, whether suppressed or burned
out, and truncates at max turns. The observation `max_turns` and the manifest
`max_steps` must match.

### benchanything.json scoring block

```json
"scoring": {
  "primary_metric": "rescue_score",
  "higher_is_better": true,
  "metrics": [
    {"name": "rescue_score",  "type": "terminal_field", "field": "norm_score",      "aggregation": "mean"},
    {"name": "lives_lost",    "type": "terminal_field", "field": "lives_lost",      "aggregation": "mean"},
    {"name": "property_lost", "type": "terminal_field", "field": "property_lost",   "aggregation": "mean"},
    {"name": "cells_burned",  "type": "terminal_field", "field": "cells_burned",    "aggregation": "mean"},
    {"name": "illegal_rate",  "type": "terminal_field", "field": "illegal_actions", "aggregation": "mean"}
  ]
}
```

The diagnostics catch the obvious exploits: a high score posted while burning huge
property, abandoning whole fronts, or spamming illegal moves. If any looks wrong,
the headline score is hollow.

---

## 7. Build phases

Guiding principle: get a random agent playing a full episode and printing a score
**before** anything is good. That surfaces integration bugs while we still have
time. The MVP is the safety net.

### Phase 0 — Setup (Saturday, first 2 to 3 hours)

Owner: everyone, together.

1. Create the public GitHub repo, add everyone as collaborators. No keys or
   secrets committed.
2. `pip install swecc-mesocosm`, run `bench init` to scaffold `adapter.py` and a
   starter `benchanything.json`. Note the real `BaseEnv` and `StepResult`
   signatures from the SDK and match them exactly.
3. Lock the section 5 and 6 mechanics into the repo as `MECHANICS.md` so nobody
   drifts on actions, spread, or scoring. Scenario parameters stay flexible.
4. Stub the file tree from section 4 with empty classes that import cleanly.

**Done when:** the repo imports without errors and `adapter.py` starts and answers
`/health` with the empty stub env.

### Phase 1 — Engine plus one multi-front fire, end to end (Saturday, rest of day)

Owners: Engine, Fire model, Scenarios in parallel.

1. Engine: `grid.py` (grid plus scenario generator), `casualty.py`, `crews.py`
   (crew state and the move/suppress/prep/evacuate effects), and the `env.py`
   `reset`/`step` loop. Wire the pre-rolled schedule.
2. Fire model: `fire.py` as a cellular automaton. Ignite a non-burning neighbor by
   comparing the pre-rolled roll against a likelihood from the neighbor's
   spreadability and the source cell's intensity, no wind. Intensity grows by the
   pre-rolled increment; a cell burns its pre-rolled timer then dies. Seed multiple
   ignition clusters so several fronts grow at once from the start.
3. Scenarios: `baselines/agents.py` with a `RandomAgent` and a `GreedyAgent`
   (greedy = send crews to the highest intensity-times-population cell they can
   reach). Write `run_local.py` to play an agent through one scenario and print the
   raw `info`.
4. Compute and hardcode the do-nothing and greedy baselines for 2 or 3 test
   scenarios so normalization returns sane numbers.

**Done when:** `python run_local.py` runs a RandomAgent through a multi-front fire
scenario start to finish and prints a `norm_score`. This is the MVP.

### Phase 2 — Mesocosm integration (Saturday night or Sunday morning)

Owner: Integration.

1. Fill in `benchanything.json`: manifest fields, `binding_vow` (observation json,
   action json with the section 5 command format described in plain language so
   the agent knows the shape, reward range, `max_steps`, `deterministic_reset:
   true`, `supports_seed: true`), and the section 6 scoring block.
2. Run the slide-7 local checklist: start `adapter.py` on a port, curl `/health`,
   `/reset` (confirm same seed gives same first observation), `/step`, `/close`.
3. Submit the repo URL through the form or API. Confirm status reaches ready.
4. Run a Test Bench against a **cheap or local model first** (DeepSeek, Ollama).
   The deck says to do this before spending mesocosm or Cursor credits. Pull one
   real trace.

**Done when:** a real model has played at least one full episode through mesocosm
and we have a trace file in hand.

### Phase 3 — Multi-front difficulty and the crew squeeze (Sunday)

Owners: Fire model, Scenarios, Engine support.

1. Fire model plus Scenarios: make the fronts differ. Vary per-front ignition
   timing, initial intensity, spreadability, and the population behind each front,
   so the fires are not interchangeable and the agent has to rank them.
2. Scenarios: build the final fixed set of 10 to 20 seeded scenarios. Include
   several where two or three fronts threaten different population centers and the
   crews cannot hold every line.
3. Engine plus Scenarios: tune crew count and action points so the crews can cover
   only roughly 50 to 70 percent of full need. Too many crews makes "do
   everything" optimal and tests nothing. The squeeze is the benchmark.
4. Recompute do-nothing and greedy baselines for every final scenario and bake them
   into the scenario configs.

**Done when:** there is a scenario where a model visibly has to let one front burn
to save a more valuable one, and the scenario set with baked baselines is
committed.

### Phase 4 — Visualization, video, and submission (Sunday night into Monday morning)

Owners: Viz lead, everyone on polish.

1. Viz: point an LLM at the trace JSON and have it build a web animation of the
   grid over time, fires spreading, crews moving and prepping and evacuating, a
   running casualty and score counter. The viz is the easy part once traces exist.
2. Record the 1 to 2 minute video: open on several fires growing at once, show the
   agent's crew choices and the front it chooses to abandon, end on the score
   versus the greedy baseline. Voiceover, no face needed. Upload as unlisted
   YouTube.
3. Write the README: what Triage measures, how to run it, the scenario set, the
   scoring rationale, and how to reuse the traces.
4. Submit on Devpost: team, repo, demo site, video.

**Done when:** all four Devpost items are attached. Target this for Monday late
morning. Leave the afternoon before 4 PM as pure buffer.

---

## 8. Timeline at a glance

- **Sat AM:** Phase 0.
- **Sat PM:** Phase 1 (MVP done by end of day).
- **Sat night / Sun AM:** Phase 2 (real trace in hand).
- **Sun:** Phase 3 (multi-front difficulty, scenarios, crews tuned).
- **Sun night / Mon AM:** Phase 4 (viz, video, README, submit).
- **Mon, by 4:00 PM:** submitted, with buffer to spare.

---

## 9. Risks and fallbacks

- **Simulator slips.** The whole project rests on the engine. If Phase 1 runs
  long, ship a scenario with fewer fronts and a smaller grid. Because board size,
  crews, and fires are all scenario dials, difficulty is a thing we turn down at
  the end, not a thing we cut. A clean two-front env beats a broken five-front one.
- **Model credits run dry.** Develop and debug entirely against local or cheap
  models. Save expensive mesocosm runs for the final scoring pass only.
- **Determinism bugs.** If two identical-seed runs diverge, the pre-rolled schedule
  is being bypassed somewhere. This is the most likely silent bug. Test it
  explicitly in Phase 2 step 2.
- **Reward hacking in the demo.** Watch the diagnostics. A high score with junk
  `property_lost`, `cells_burned`, or `illegal_actions` is not a real result, and a
  judge may ask.

---

## 10. Two traps we keep reminding each other about

1. **Keep the ignition rolls hidden.** The agent sees spreadability, which is the
   risk level a firefighter reads off the terrain and can lower by prepping, but it
   never sees the per-cell roll that decides which at-risk cell actually catches.
   The skill is planning against risk, not reading outcomes off the screen.
2. **Action effects stay discrete and integer.** The moment an effect goes
   continuous the verifier gets mushy and scope explodes. Crew size times an
   integer constant, intensity on a 1 to 100 scale, whole people evacuated.
