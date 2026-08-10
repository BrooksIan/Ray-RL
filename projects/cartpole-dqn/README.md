# CartPole DQN

Train **Deep Q-Networks (DQN)** on Gymnasium [`CartPole-v1`](https://gymnasium.farama.org/environments/classic_control/cart_pole/) with the current Ray RLlib API stack.

Companion to the cover [Taxi PPO](../taxi-ppo/) demo: same Workbench-friendly loop, but **off-policy** learning with a replay buffer.

| | |
| --- | --- |
| Entrypoint | `train_cartpole_dqn.py` |
| Algorithm | DQN (`DQNConfig`) |
| Env | `CartPole-v1` |
| Workbench guide | [docs/getting-started.md](../../docs/getting-started.md) |

## The problem: what CartPole is asking the agent to do

A cart moves left/right on a track while balancing an inverted pole:

1. Each step the agent applies a **discrete** force: left or right.
2. The pole angle and cart position evolve under physics.
3. The episode ends when the pole falls too far, the cart leaves the track, or a time limit is hit.
4. Reward is **+1 per step survived** — longer balance = higher return.

So the job is closed-loop **stabilization**: keep a dynamic system inside safe bounds, not classify a static row of features.

## Why RL / why DQN

| Approach | Why it falls short |
| --- | --- |
| Supervised learning | Needs labeled “correct push” at every state; you only get survival reward after acting. |
| Open-loop script | One fixed sequence of left/right fails as soon as the pole starts tipping differently. |

**RL** learns a reactive policy from trial and error. **DQN** is a strong fit here because:

- Actions are **discrete** (DQN’s native setting — unlike continuous torque problems better suited to SAC).
- Learning is **off-policy**: a replay buffer reuses past `(s, a, r, s')` transitions, which is sample-efficient and a clear contrast to on-policy PPO on Taxi.
- CartPole is small enough for a laptop / Workbench CPU session.

Enterprise analogy: discrete control of a process (on/off, left/right actuators) where you care about keeping the system stable over time.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r projects/cartpole-dqn/requirements.txt

python projects/cartpole-dqn/train_cartpole_dqn.py
```

Or from this directory:

```bash
cd projects/cartpole-dqn
pip install -r requirements.txt
python train_cartpole_dqn.py
```

## What it does

1. Configure DQN on `CartPole-v1` with one EnvRunner
2. Use a prioritized episode replay buffer (off-policy reuse)
3. Train for ten iterations, evaluate, then `algo.stop()`

## Expected output

Mean episode return should trend upward (CartPole is “solved” around a mean return of 475; a short smoke run may land lower but should improve). Example from a local 10-iter run:

```text
iter=1   episode_return_mean=20.5   env_steps=1000.0
iter=5   episode_return_mean=84.1   env_steps=5000.0
iter=10  episode_return_mean=257.2  env_steps=10000.0
evaluate episode_return_mean=305.0
```

Ray may still print internal deprecation notices; they are harmless for this smoke test.

## Architecture

```text
DQN Algorithm + Torch Q-network (RLModule)
        │ sample
        ▼
EnvRunner ── CartPole-v1
        │ store / replay
        ▼
PrioritizedEpisodeReplayBuffer ──► Learner update
        │
        ▼
Eval EnvRunner (manual algo.evaluate())
```

## Requirements

- Python 3.10–3.12
- See [`requirements.txt`](requirements.txt) (`ray[rllib]==2.56.1`, Torch, Gymnasium)

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).

## See also

- Cover demo: [Taxi PPO](../taxi-ppo/) (on-policy, discrete logistics)
- [RLlib DQN docs](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#dqn)
