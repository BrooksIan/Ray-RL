# Pendulum SAC

Train **Soft Actor-Critic (SAC)** on Gymnasium [`Pendulum-v1`](https://gymnasium.farama.org/environments/classic_control/pendulum/) with the current Ray RLlib API stack.

Completes the cover ladder with **continuous** control: [Taxi PPO](../taxi-ppo/) (discrete logistics) → [CartPole DQN](../cartpole-dqn/) (discrete control) → **Pendulum SAC** (continuous torque).

| | |
| --- | --- |
| Entrypoint | `train_pendulum_sac.py` |
| Notebook | [`pendulum_sac.ipynb`](pendulum_sac.ipynb) |
| Algorithm | SAC (`SACConfig`) |
| Env | `Pendulum-v1` |
| Workbench guide | [docs/getting-started.md](../../docs/getting-started.md) |

## The problem: what Pendulum is asking the agent to do

A rigid pendulum swings under gravity. Each step the agent applies a **continuous torque** (a real-valued action in a bounded range) to the free end:

1. Observations describe angle (via `cos` / `sin`) and angular velocity.
2. The agent outputs a continuous torque command — not a left/right menu.
3. Reward penalizes angle from upright, angular velocity, and torque magnitude (energy).
4. Goal: swing up and **hold** the pendulum near vertical with efficient control.

So the job is **actuator / continuous control**: choose how hard to push, not which discrete button to press.

## Why RL / why SAC

| Approach | Why it falls short |
| --- | --- |
| Supervised learning | Needs labeled “correct torque” every timestep; you only get a cost after acting. |
| Discrete DQN | Action space is continuous; discretizing torque poorly scales and loses precision. |
| Fixed open-loop torque schedule | Fails across start angles and velocities; you need a reactive policy. |

**RL** learns a feedback policy from costs over time. **SAC** fits because:

- It targets **continuous** action spaces with stochastic, entropy-regularized exploration.
- It is **off-policy** (replay buffer), sample-efficient for control tasks.
- Pendulum is the standard RLlib SAC sandbox ([docs tuned example](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac)).

Enterprise analogy: motors, valves, setpoints — anywhere the control signal is a continuous effort, not a small discrete menu.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r projects/pendulum-sac/requirements.txt

python projects/pendulum-sac/train_pendulum_sac.py
```

Or from this directory:

```bash
cd projects/pendulum-sac
pip install -r requirements.txt
python train_pendulum_sac.py
```

Notebook twin: [`pendulum_sac.ipynb`](pendulum_sac.ipynb) (select the same venv kernel; ~5–10 min).

## What it does

1. Configure SAC on `Pendulum-v1` with one EnvRunner
2. Use a prioritized episode replay buffer (off-policy reuse)
3. Train for fifteen iterations, evaluate, then `algo.stop()`

## Expected output

Pendulum returns are **negative** (cost). Learning means the mean becomes **less negative** over time. Example from a local 15-iter run (~9 minutes on CPU):

```text
iter=1   episode_return_mean=-1335.8  env_steps=1000.0
iter=5   episode_return_mean=-250.6   env_steps=5000.0
iter=9   episode_return_mean=-97.2    env_steps=9000.0
iter=15  episode_return_mean=-162.5   env_steps=15000.0
evaluate episode_return_mean=-120.8
```

Strong policies often land near **−200** or better; variance between iters is normal. Ray may still print internal deprecation notices; they are harmless for this smoke test.

## Architecture

```text
SAC Algorithm + Torch actor/critic (RLModule)
        │ sample
        ▼
EnvRunner ── Pendulum-v1 (continuous torque)
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

- [Taxi PPO](../taxi-ppo/) — on-policy, discrete logistics
- [CartPole DQN](../cartpole-dqn/) — off-policy, discrete control
- [RLlib SAC docs](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#sac)
