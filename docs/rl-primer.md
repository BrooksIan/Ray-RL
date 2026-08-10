# Reinforcement learning primer

Short vocabulary for readers new to RL. Used by the cover Taxi PPO demo and any future projects in this blueprint.

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
| Action space | All possible actions |
| State space | Complete environment description (nothing hidden) |
| Observation space | What the agent actually sees |
| Reward | Feedback after each action |
| Episode | Sequence from initial observation to terminal state |
| Policy (π) | Mapping observations → actions; often a neural net in deep RL |

![RL with policy](../assets/RLWithPolicy.png)

> In RL, “model” is roughly equivalent to “policy,” but policy is more specific because it is trained in a particular environment. For deployment, people often say “model” in the usual ML sense.

## Why Ray RLlib

[RLlib](https://docs.ray.io/en/latest/rllib/index.html) provides production-oriented, distributed RL algorithms (PPO, DQN, SAC, and others) with a unified config/train/evaluate API. This blueprint uses the **new API stack**: EnvRunners for sampling, RLModules for policies, and connectors (for example `FlattenObservations` on discrete Taxi states).

## Further reading

- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html)
- [Proximal Policy Optimization (paper)](https://arxiv.org/abs/1707.06347)
- [Deep RL: Pong from pixels](http://karpathy.github.io/2016/05/31/rl/)
