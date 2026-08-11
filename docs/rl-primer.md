# Reinforcement learning primer

Short vocabulary for readers new to RL, plus how this blueprint’s projects map onto common RLlib patterns.

## The loop

One or more **agents** interact with an **environment** (simulator or real sensors/actuators). At each step:

1. The agent receives an **observation**
2. The agent selects an **action**
3. The environment returns a **reward** and the next observation
4. Episodes end when the environment reaches a terminal state (success, failure, or timeout)

The learning system trains a **policy** — which actions maximize long-term cumulative reward.

![RL overview](../assets/RLOverview.png)

## Terminology

| Term | Meaning |
| --- | --- |
| Action space | All possible actions (discrete menu or continuous values) |
| State space | Complete environment description (nothing hidden) |
| Observation space | What the agent actually sees |
| Reward | Feedback after each action |
| Episode | Sequence from initial observation to terminal state |
| Policy (π) | Mapping observations → actions; often a neural net in deep RL |
| On-policy | Learn only from data collected by the current policy (e.g. PPO) |
| Off-policy | Reuse past transitions from a replay buffer (e.g. DQN, SAC) |
| Multi-agent | Several agents/policies act in one environment loop |
| Offline / imitation | Learn from logged trajectories without online exploration (e.g. BC, MARWIL) |
| Custom environment | Your `gymnasium.Env` registered with Ray (`register_env`) — not only stock Gym |

![RL with policy](../assets/RLWithPolicy.png)

> In RL, “model” is roughly equivalent to “policy,” but policy is more specific because it is trained in a particular environment. For deployment, people often say “model” in the usual ML sense.

## How the blueprint projects fit

| Project | Env | Algorithm | Idea |
| --- | --- | --- | --- |
| [Taxi PPO](../projects/taxi-ppo/README.md) | `Taxi-v3` | PPO | Delayed-reward pickup/delivery; on-policy discrete logistics |
| [CartPole DQN](../projects/cartpole-dqn/README.md) | `CartPole-v1` | DQN | Keep a system stable with discrete actions; off-policy + replay |
| [Pendulum SAC](../projects/pendulum-sac/README.md) | `Pendulum-v1` | SAC | Continuous torque — actuators, not button menus |
| [Multi-Agent CartPole](../projects/multiagent-cartpole/README.md) | `MultiAgentCartPole` | Multi-agent PPO | Fleet of controllers; policies + mapping function |
| [Offline BC / MARWIL](../projects/offline-marwil/README.md) | Logged `CartPole-v1` | BC (MARWIL-ready) | Learn from trajectories when online exploration is costly |
| [Custom playground PPO](../projects/custom-env-ppo/README.md) | `TicketQueue-v0` (custom) | PPO | Define your own Gymnasium env, then train |

### Concrete example: Taxi (cover)

Taxi is a pickup-and-delivery task: drive to a passenger, pick them up, drive to their destination, drop them off. Reward is mostly delayed (`+20` only on success, `-1` per step).

That is why RL is the natural approach — you learn a **sequence of decisions** from outcome scores, not fitting labels. See the [blueprint Use Case](../README.md#use-case) for a fuller comparison.

### Concrete example: custom playground (`TicketQueue-v0`)

Stock Gym envs teach algorithms. A **custom env** teaches how to encode *your* process: observations, actions, and rewards. Step 6 ([custom-env-ppo](../projects/custom-env-ppo/README.md)) defines a single-server ticket queue, registers it with `register_env`, and trains PPO — the same trainer as Taxi on dynamics you control. For a step-by-step use-case → playground checklist, see [How to define a playground for your use case](../projects/custom-env-ppo/README.md#how-to-define-a-playground-for-your-use-case).

## Why Ray RLlib

[RLlib](https://docs.ray.io/en/latest/rllib/index.html) provides production-oriented, distributed RL algorithms (PPO, DQN, SAC, BC, MARWIL, and others) with a unified config/train/evaluate API. This blueprint uses the **new API stack**: EnvRunners for sampling, RLModules for policies, connectors (for example `FlattenObservations` on discrete Taxi states), `.multi_agent(...)` for fleets, `.offline_data(...)` for logged Parquet, and `register_env` for custom Gymnasium playgrounds.

Algorithm catalog: [RLlib algorithms](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html).

## Platform extras (Cloudera AI)

| Topic | Doc |
| --- | --- |
| Workbench session + multi-worker EnvRunners | [deploy/README.md](../deploy/README.md) |
| Optional MLflow episode-return tracking | [deploy/mlflow.md](../deploy/mlflow.md) |
| Bring-your-own offline Parquet | [offline-marwil README](../projects/offline-marwil/README.md#bring-your-own-parquet-logs) |
| Custom Gymnasium playground + PPO | [custom-env-ppo README](../projects/custom-env-ppo/README.md) |
| Notebook twins for every project | [docs/README.md](README.md) |

## Further reading

- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html)
- [RLlib offline RL](https://docs.ray.io/en/latest/rllib/rllib-offline.html)
- [Gymnasium Env API](https://gymnasium.farama.org/api/env/)
- [Proximal Policy Optimization (paper)](https://arxiv.org/abs/1707.06347)
- [Deep RL: Pong from pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Project index](../projects/README.md)
