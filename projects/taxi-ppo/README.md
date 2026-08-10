# Taxi PPO

Train **Proximal Policy Optimization (PPO)** on Gymnasium [`Taxi-v3`](https://gymnasium.farama.org/environments/toy_text/taxi/) with the current Ray RLlib API stack.

This is the **cover demo** for the [Ray RLlib on Cloudera AI](../../README.md) blueprint.

| | |
| --- | --- |
| Entrypoint | `train_taxi_ppo.py` |
| Notebook | `RayRLTest.ipynb` |
| Workbench guide | [docs/getting-started.md](../../docs/getting-started.md) |
| Catalog metadata | [`METADATA.yaml`](../../METADATA.yaml) (`featured_project`) |

## The problem: what Taxi is asking the agent to do

Taxi is a miniature **pickup-and-delivery** job on a 5×5 city grid:

```text
+---------+
|R: | : :G|
| : | : : |
| : : : : |
| | : | : |
|Y| : |B: |
+---------+
```

- **R / G / Y / B** are the only legal passenger stands and destinations.
- Each episode samples a taxi start, a passenger location, and a destination.
- The agent must: navigate → **pickup** → navigate → **drop-off**.
- **Actions (6):** move south / north / east / west, pickup, drop-off.
- **Rewards:** `+20` correct delivery, `-1` each step, `-10` illegal pickup/drop-off.

So the “problem being solved” is not predicting a label. It is learning a **driving/dispatch policy**: from any valid situation, choose moves that deliver the passenger quickly and legally.

In enterprise terms, that maps to sequential logistics — get an asset to a job, complete the handoff, finish at the right place — where success is measured over a whole trajectory.

## Why RL (not supervised learning) 

Supervised models shine when you already have `(state → correct action)` examples. Taxi does not give you that. It gives you an environment and a score. The right action now depends on passenger location, destination, and whether they are already in the cab — and the big bonus arrives only at the end.

That is exactly when RL is the better tool:

1. **Delayed reward** — delivery credit is sparse; step cost teaches efficiency.
2. **Many scenarios, one policy** — hundreds of start configurations; you want generalization, not one hard-coded path.
3. **Interactive feedback** — illegal pickups and wasted steps are experienced, then avoided.
4. **Same pattern as real control** — fleet routing, warehouse robots, and ops playbooks grow from this loop; Taxi is the readable sandbox.

Could you hand-write rules for this tiny map? Yes. That misses the point of the demo: learn a policy from rewards with the same RLlib stack you will use when the simulator is too large to script.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r projects/taxi-ppo/requirements.txt

python projects/taxi-ppo/train_taxi_ppo.py
```

Or from this directory:

```bash
cd projects/taxi-ppo
pip install -r requirements.txt
python train_taxi_ppo.py
```

Notebook twin: [`RayRLTest.ipynb`](RayRLTest.ipynb) (select the same venv kernel).

## What it does

1. Configure PPO on `Taxi-v3` with two EnvRunners
2. Flatten discrete Taxi observations with `FlattenObservations`
3. Train for five iterations, evaluate, then `algo.stop()`

## Expected output

Mean episode return should improve across iterations (still negative for a short Taxi run), then a final evaluate line, for example:

```text
iter=1  episode_return_mean=-747.9  env_steps=4000.0
…
iter=5  episode_return_mean=-410.6  env_steps=20000.0
evaluate  episode_return_mean=-353.9
```

Ray may still print internal deprecation notices; they are harmless for this smoke test.

## Architecture

```text
PPO Algorithm + Torch RLModule
        │ sample / sync
        ▼
EnvRunners (×2) ── Taxi-v3 ── FlattenObservations
        │
        ▼
Eval EnvRunner (manual algo.evaluate())
```

## Requirements

- Python 3.10–3.12
- See [`requirements.txt`](requirements.txt) (`ray[rllib]==2.56.1`, Torch, Gymnasium)

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).

## See also

Next steps in the blueprint ladder:

- [CartPole DQN](../cartpole-dqn/) — off-policy discrete control
- [Pendulum SAC](../pendulum-sac/) — continuous actuators
- [Multi-Agent CartPole PPO](../multiagent-cartpole/) — fleet / policy mapping
- [Project index](../README.md) · [Cover README](../../README.md)
