# Custom playground + PPO (`TicketQueue-v0`)

Define **your own Gymnasium environment**, register it with Ray, and train **PPO** — companion step 6 in the blueprint ladder.

This is the “bring your own decision process” lab: same RLlib stack as Taxi, but the playground is a small **ticket / job queue** you can read and change.

| | |
| --- | --- |
| Entrypoint | `train_queue_ppo.py` |
| Env definition | [`queue_env.py`](queue_env.py) |
| Notebook | [`custom_env_ppo.ipynb`](custom_env_ppo.ipynb) |
| Algorithm | PPO (`PPOConfig`) |
| Env | `TicketQueue-v0` (custom) |

## The problem

A single operator (server) faces a backlog of tickets:

1. Tickets **arrive** stochastically each step.
2. Each ticket needs a few units of **work**.
3. Each step the agent chooses **idle** or **work** on the head ticket.
4. A full queue that rejects an arrival is an **SLA / overflow** miss.

Rewards encode ops pressure: pay for backlog every step, get a bonus when a ticket completes, and take penalties for working an empty queue or overflowing capacity.

Enterprise analogy: support desk, batch job scheduler, or plant work-order queue — sequential decisions under arrival uncertainty, not a labeled “correct action” dataset.

## Why a custom env (and why RL)

| Approach | Why it falls short |
| --- | --- |
| Supervised learning | No oracle labels for idle vs work at every backlog state |
| Fixed “always work” heuristic | Decent baseline here, but the point is learning the loop on *your* dynamics / costs |
| Stock Gym only | Great for algorithms; weak for “this is our process” storytelling |

**RL** fits because return is delayed and stateful (drain now vs risk overflow later). **Custom Gymnasium** is how you encode that process for RLlib.

## How to define a playground for your use case

A playground is a **simplified simulator** of the decisions you care about — not a full digital twin. Start from the business story, then map four contracts Gymnasium requires: observation, action, reward, and episode end.

### 1. Write the use case in one paragraph

Answer:

1. **Who decides?** (dispatcher, operator, controller, allocator)
2. **What can they do each step?** (assign, idle, buy, route, …)
3. **What do they see?** (queue length, inventory, location, SLA clock)
4. **What makes an outcome good or bad?** (complete jobs, avoid overflow, reduce wait, hit a target)
5. **When does a “shift” / episode end?** (fixed horizon, empty queue, mission complete)

If you cannot answer those, the env will be fuzzy and hard to learn.

**TicketQueue answers:** a single server decides idle vs work; sees normalized backlog / head work / time / at-capacity; wants completions without backlog or SLA overflow; episode is a fixed 40-step shift.

### 2. Map use case → Gymnasium contracts

| Contract | Question | TicketQueue choice | Tips |
| --- | --- | --- | --- |
| **Observation** | What is available *before* the action? | Queue length, head remaining work, time fraction, at-capacity flag | Prefer small, numeric features the policy can reuse; normalize to ~\[0, 1\] when using a Box |
| **Action space** | What mutually exclusive choices exist each step? | `Discrete(2)`: idle / work | Start discrete and tiny; add actions only when the story needs them |
| **Transition** | How does the world change after an action? | Work decrements head ticket; Bernoulli arrivals; overflow if full | Keep stochasticity simple (coin flips, small Uniforms) |
| **Reward** | How do you score *this* step toward the business goal? | −backlog, +complete, −illegal work, −overflow | Align signs with “higher return = better ops”; avoid huge sparse bonuses until the smoke run works |
| **Termination** | When does the episode stop? | Truncate at `episode_len` | Fixed horizon is easiest for first playgrounds |

Implement those in a `gymnasium.Env` subclass: `reset()` → first obs; `step(action)` → `(obs, reward, terminated, truncated, info)`.

### 3. Keep the first playground deliberately small

| Do | Avoid (at first) |
| --- | --- |
| 2–6 actions | Continuous actuators + images + multi-agent together |
| Observation dim ≤ ~10 | Full ERP / ticket text dumps |
| Episode tens to low hundreds of steps | Day-long sims before learning works |
| One clear cost + one clear success signal | Ten competing KPIs with unclear tradeoffs |
| Illegal actions → penalty (or later: masking) | Silent no-ops that confuse credit assignment |

You can always add fidelity after PPO shows an upward `episode_return_mean` curve.

### 4. Wire the playground into RLlib

1. Put the env in a module (here: [`queue_env.py`](queue_env.py)).
2. Expose a factory: `def make_…(config): return MyEnv(**config)`.
3. `register_env("MyEnv-v0", make_…)`.
4. `PPOConfig().environment("MyEnv-v0")` — same as Taxi, different name.

That is the whole “custom playground” pattern this project teaches.

### 5. Validate before long training

1. **Manual rollouts** — random or scripted actions; print `obs`, `reward`, `info` (notebook cell 1).
2. **Sanity baselines** — “always work” / “always idle” should score differently if the reward is meaningful.
3. **Short PPO smoke** — 10–15 iters; expect return to move in the right direction.
4. **Tune the story, not only the algorithm** — if learning stalls, revisit reward scale and observation clarity before changing PPO hyperparameters.

### Worked mapping: ops queue → `TicketQueue-v0`

```text
Use case                         Playground field
─────────────────────────────    ─────────────────────────────
Support / job backlog            queue length (+ head work left)
Operator works or waits          action: work | idle
Backlog costs money              reward: -queue_cost * len(queue)
Finishing a ticket is good       reward: +complete_bonus
Queue full → missed SLA          reward: -overflow_penalty
One shift of decisions           episode_len steps (truncated)
```

Change the knobs in `TicketQueueEnv.__init__` (`arrival_prob`, `queue_cost`, `overflow_penalty`, …) to retarget the same skeleton toward “busier arrivals,” “stricter SLA,” or “cheaper backlog.”

### Other use cases you can sketch the same way

| Use case sketch | Obs (examples) | Actions (examples) | Reward sketch |
| --- | --- | --- | --- |
| Inventory reorder | Stock level, demand signal, days to delivery | Wait / place order | −holding −stockout; −order cost |
| Simple router / taxi-like | Position, job pending, destination flag | Move N/E/S/W, pickup/drop | −step; +delivery; −illegal |
| Energy / thermostat | Temp error, outdoor proxy | Heat / cool / off | −\|error\|; −energy use |
| Batch GPU / job packer | Queue depth, free slots | Admit job A/B / wait | +utilization; −wait; −reject |

Each row becomes one env class + `register_env` + the existing PPO script pattern.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r projects/custom-env-ppo/requirements.txt

python projects/custom-env-ppo/train_queue_ppo.py
```

Notebook twin: [`custom_env_ppo.ipynb`](custom_env_ppo.ipynb).

Optional MLflow / workers (same pattern as Taxi):

```bash
export RAY_RL_MLFLOW=1
export MLFLOW_TRACKING_URI=sqlite:///./mlflow.db
python projects/custom-env-ppo/train_queue_ppo.py --num-env-runners 2 --train-iters 12
```

## What success looks like

Random / early policies often sit around low (negative) returns from backlog + illegal work. Over ~12 iters, `episode_return_mean` should **climb** (less backlog, more completions). Example local smoke run:

```text
iter=1   episode_return_mean=-52.9
iter=6   episode_return_mean=-22.9
iter=12  episode_return_mean=-17.5
evaluate episode_return_mean=-13.1
```

Exact numbers vary with seed; look for a clear upward trend, then an `evaluate` line.

## How registration works

```text
queue_env.TicketQueueEnv  (gymnasium.Env)
        │
        ▼ register_env("TicketQueue-v0", make_ticket_queue_env)
PPOConfig().environment("TicketQueue-v0")
        │
        ▼
EnvRunners sample ──► PPO Learner ──► evaluate()
```

Edit rewards, arrival probability, or observation features in [`queue_env.py`](queue_env.py), then re-run training — that is the playground loop.

## Architecture

```text
PPO Algorithm + Torch RLModule
        │
        ▼
EnvRunners ── TicketQueue-v0 (custom Gymnasium)
        │
        ▼
Eval EnvRunner (manual algo.evaluate())
```

## Requirements

- Python 3.10–3.12
- [`requirements.txt`](requirements.txt): `ray[rllib]`, Torch, Gymnasium

## See also

- [Taxi PPO](../taxi-ppo/) — same algorithm on a stock Gym env
- [Offline BC](../offline-marwil/) — next: log this env and learn offline
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Project index](../README.md) · [Cover README](../../README.md)
