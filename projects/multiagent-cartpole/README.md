# Multi-Agent CartPole PPO

Train **multi-agent PPO** on RLlib’s `MultiAgentCartPole` (two carts balancing poles in one environment) with the current Ray RLlib API stack.

This project highlights RLlib’s multi-agent surface: **policies**, **policy mapping**, and per-agent returns — the “fleet / coordinated controllers” step after single-agent Taxi, CartPole, and Pendulum.

| | |
| --- | --- |
| Entrypoint | `train_multiagent_cartpole.py` |
| Algorithm | PPO (`PPOConfig`) + `.multi_agent(...)` |
| Env | `MultiAgentCartPole` (`num_agents=2`) |
| Workbench guide | [docs/getting-started.md](../../docs/getting-started.md) |

## The problem: what multi-agent CartPole is asking

`MultiAgentCartPole` wraps classic CartPole so **several agents** each control their own cart/pole:

1. Each agent observes its own cart/pole state.
2. Each agent chooses left/right force each step.
3. Episodes end when agents fall (per MultiAgentCartPole rules).
4. Reward is still survival — but now you train **a set of policies** that act together in one env loop.

Enterprise analogy: a **fleet of controllers** (robots, HVAC zones, network nodes) that share a simulation/runtime but may use distinct policies.

## Why RL / why multi-agent PPO

| Approach | Why it falls short |
| --- | --- |
| One single-agent script copied N times | Misses joint rollouts, shared sampling, and RLlib policy mapping |
| Centralized supervised labels | No oracle action per agent/state; feedback is interactive return |
| Hard-coded sync rules | Do not adapt as dynamics or agent counts change |

**Multi-agent PPO** fits because RLlib already supports:

- Declaring multiple policies (`p0`, `p1`, …)
- Mapping `agent_id → policy_id` with `policy_mapping_fn`
- Training those policies from the same EnvRunner loop

This demo uses **separate policies** (distinct controllers). For a shared fleet brain, map every agent to one policy id instead.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r projects/multiagent-cartpole/requirements.txt

python projects/multiagent-cartpole/train_multiagent_cartpole.py
```

Or from this directory:

```bash
cd projects/multiagent-cartpole
pip install -r requirements.txt
python train_multiagent_cartpole.py
```

## What it does

1. Register `MultiAgentCartPole` with 2 agents
2. Configure PPO with policies `p0` / `p1` and a mapping fn
3. Train for ten iterations, print per-agent (or per-policy) returns, evaluate, then `algo.stop()`

## Expected output

Episode return (summed multi-agent survival signal) should trend upward. Example from a local 10-iter run (~40s):

```text
iter=1   episode=47.9   env_steps=4000.0
iter=5   episode=184.1  env_steps=20000.0
iter=10  episode=395.8  env_steps=40000.0
evaluate episode=415.4
```

When RLlib exposes per-agent / per-policy means, the script also prints those. Ray may still print internal deprecation notices; they are harmless for this smoke test.

## Architecture

```text
PPO Algorithm
  policies: p0, p1   (policy_mapping_fn: agent_id -> p{agent_id})
        │ sample / sync
        ▼
EnvRunners ── MultiAgentCartPole (2 agents)
        │
        ▼
Eval EnvRunner (manual algo.evaluate())
```

## Requirements

- Python 3.10–3.12
- See [`requirements.txt`](requirements.txt) (`ray[rllib]==2.56.1`, Torch, Gymnasium)

`MultiAgentCartPole` comes from Ray’s bundled examples package (`ray.rllib.examples.envs...`) shipped with `ray[rllib]`.

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).

## See also

- [CartPole DQN](../cartpole-dqn/) — single-agent discrete control (off-policy)
- [Taxi PPO](../taxi-ppo/) — single-agent on-policy cover demo
- [RLlib multi-agent docs](https://docs.ray.io/en/latest/rllib/rllib-env.html#multi-agent-and-hierarchical)
