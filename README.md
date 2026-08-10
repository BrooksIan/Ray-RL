# Cloudera Blueprint: Ray RLlib on Cloudera AI

> Multi-project **Ray RLlib** examples for Cloudera AI Workbench. The cover demo trains **PPO on Gymnasium Taxi-v3**; companions add DQN, SAC, and multi-agent PPO. Catalog fields: [`METADATA.yaml`](METADATA.yaml).

## Table of Contents

- [Overview](#overview)
- [Learning path](#learning-path)
- [Demo](#demo)
- [Use Case](#use-case)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Software Components](#software-components)
- [Target Audience](#target-audience)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Documentation](#documentation)
- [Projects](#projects)

## Overview

**Ray RLlib on Cloudera AI** helps ML engineers and architects learn distributed reinforcement learning where they already work — inside Cloudera AI Workbench. The blueprint ships a featured PPO → Taxi-v3 quickstart on RLlib’s current EnvRunner / RLModule API, then a ladder of companion projects that teach off-policy discrete control (DQN), continuous actuators (SAC), multi-agent fleets (multi-agent PPO), and offline learning from logs (MARWIL). Cloudera value: run Ray-based RL experiments on the same governed AI workbench used for notebooks, sessions, and team collaboration.

## Learning path

Work through the projects in order — each adds one RLlib capability without leaving Gymnasium-scale demos:

| Step | Project | Algorithm | What you learn |
| --- | --- | --- | --- |
| 1 (cover) | [`taxi-ppo`](projects/taxi-ppo/) | PPO | On-policy RL; delayed-reward logistics / dispatch |
| 2 | [`cartpole-dqn`](projects/cartpole-dqn/) | DQN | Off-policy + replay; discrete process control |
| 3 | [`pendulum-sac`](projects/pendulum-sac/) | SAC | Continuous torque / actuator control |
| 4 | [`multiagent-cartpole`](projects/multiagent-cartpole/) | Multi-agent PPO | Policies, mapping fn, fleet-style controllers |
| 5 | [`offline-marwil`](projects/offline-marwil/) | MARWIL | Learn from logged trajectories (no online explore) |

All share the same install pattern: `pip install -r projects/<slug>/requirements.txt` then run the project entrypoint.

## Demo

Primary demo (no Reprise recording required):

1. Follow [Quickstart](#quickstart) or [Workbench getting started](docs/getting-started.md).
2. Run:

   ```bash
   python projects/taxi-ppo/train_taxi_ppo.py
   ```

3. Confirm episode return improves across five iterations, then an `evaluate` line prints.

Interactive twins (same venv / `requirements.txt` as each project):

| Project | Notebook |
| --- | --- |
| Taxi PPO (cover) | [`projects/taxi-ppo/RayRLTest.ipynb`](projects/taxi-ppo/RayRLTest.ipynb) |
| CartPole DQN | [`projects/cartpole-dqn/cartpole_dqn.ipynb`](projects/cartpole-dqn/cartpole_dqn.ipynb) |
| Pendulum SAC | [`projects/pendulum-sac/pendulum_sac.ipynb`](projects/pendulum-sac/pendulum_sac.ipynb) |
| Multi-Agent CartPole | [`projects/multiagent-cartpole/multiagent_cartpole.ipynb`](projects/multiagent-cartpole/multiagent_cartpole.ipynb) |
| Offline BC | [`projects/offline-marwil/offline_bc.ipynb`](projects/offline-marwil/offline_bc.ipynb) |

Sample output and troubleshooting: [`projects/taxi-ppo/README.md`](projects/taxi-ppo/README.md).

Optional catalog field: set `reprise_link` in [`METADATA.yaml`](METADATA.yaml) when a recorded walkthrough is published.

### Companion demos

```bash
python projects/cartpole-dqn/train_cartpole_dqn.py
python projects/pendulum-sac/train_pendulum_sac.py          # ~5–10 min on CPU
python projects/multiagent-cartpole/train_multiagent_cartpole.py
python projects/offline-marwil/run_pipeline.py              # record logs + offline BC
python projects/taxi-ppo/train_taxi_ppo.py --num-env-runners 4   # larger Workbench session
```

Platform extras: [multi-worker sizing](deploy/README.md#multi-worker-ray-on-workbench) · [MLflow](deploy/mlflow.md) · [BYO Parquet offline](projects/offline-marwil/README.md#bring-your-own-parquet-logs)

## Use Case

### What the Taxi problem solves

[Gymnasium Taxi](https://gymnasium.farama.org/environments/toy_text/taxi/) is a small **dispatch-and-delivery** puzzle that stands in for real sequential logistics:

1. A taxi starts somewhere on a 5×5 grid (with walls).
2. A passenger waits at one of four stands (Red / Green / Yellow / Blue).
3. The agent must **drive to the passenger**, **pick them up**, **drive to their destination**, and **drop them off**.
4. The episode ends on a successful drop-off (or after a time limit).

Rewards encode the business goal: **+20** for a correct delivery, **−1** per time step (faster routes win), and **−10** for illegal pickup/drop-off. There is no labeled dataset of “correct moves” — only this scoreboard after each action.

That is the same shape as many enterprise control problems: route a resource, complete a multi-step job, and optimize long-horizon return under constraints — not classify a single row of features.

### Why reinforcement learning is the right fit

| Approach | Why it falls short here |
| --- | --- |
| Supervised learning | Needs labeled “best action” per state. In Taxi (and real dispatch), you rarely have that oracle; you only have outcomes after a sequence of decisions. |
| Fixed rules / one shortest path | Can hard-code this tiny map, but every new passenger/destination pair is a different task. You want one **policy** that generalizes across ~300 start configurations, not a single scripted route. |
| Open-loop planning only | Works when the world is tiny and static. RL is how you *learn* a policy from interaction when the state space, stochasticity, or environment complexity grows (fleet size, traffic, custom sims). |

RL fits because Taxi is a **sequential decision** problem with **delayed credit**: the valuable +20 arrives only after a chain of moves. Algorithms like PPO improve a policy by trial and error — try routes, take the penalties and bonuses, and reinforce behaviors that raise cumulative return.

Taxi is intentionally small (500 discrete states, 6 actions), so you can see learning quickly on a laptop or Workbench session. Companion projects extend the same RLlib loop to discrete control (CartPole/DQN), continuous actuators (Pendulum/SAC), and multi-agent fleets (MultiAgentCartPole/PPO).

### Blueprint outcome

Teams leave with a reproducible first success on Cloudera AI — train and evaluate PPO on Taxi-v3 in minutes — then a clear ladder into off-policy, continuous, and multi-agent RLlib patterns under one repo.

## Key Features

- Featured **Taxi PPO** cover demo on the current RLlib API stack
- Companion projects: **DQN**, **SAC**, **multi-agent PPO**, and **offline MARWIL**
- **Multi-project** layout (`projects/<slug>/`) with a documented learning path
- Runnable scripts (plus Taxi notebook) with per-project pinned dependencies
- **Workbench-ready** deploy and sizing guidance
- **RL primer** and diagrams for non-RL practitioners
- Catalog-ready [`METADATA.yaml`](METADATA.yaml) (Apache-2.0)

## Quickstart

1. Clone the repository:

   ```bash
   git clone https://github.com/BrooksIan/Ray-RL.git
   cd Ray-RL
   ```

2. Create and activate a virtual environment (Python **3.10–3.12**):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. Install and run the cover project:

   ```bash
   pip install -r projects/taxi-ppo/requirements.txt
   python projects/taxi-ppo/train_taxi_ppo.py
   ```

4. Or open [`projects/taxi-ppo/RayRLTest.ipynb`](projects/taxi-ppo/RayRLTest.ipynb) with that venv kernel.

5. (Optional) Continue the ladder:

   ```bash
   pip install -r projects/cartpole-dqn/requirements.txt
   python projects/cartpole-dqn/train_cartpole_dqn.py
   ```

**Cloudera AI Workbench:** start a Python session (≥2 vCPU / 8 GB), clone or open this repo, then run the same install + entrypoint steps. Full guide: [docs/getting-started.md](docs/getting-started.md).

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).

## Software Components

Cover demo (Taxi PPO):

```text
Cloudera AI Workbench (session / notebook)
        │
        ▼
┌───────────────────┐     sample      ┌────────────────────┐
│  Ray RLlib PPO    │◄───────────────►│ EnvRunners (×2)    │
│  + Torch RLModule │                 │ Gymnasium Taxi-v3  │
└─────────┬─────────┘                 │ FlattenObservations│
          │ evaluate()                └────────────────────┘
          ▼
┌───────────────────┐
│ Eval EnvRunner    │
└───────────────────┘
```

| Component | Role |
| --- | --- |
| Cloudera AI Workbench | Runtime for sessions, notebooks, and team projects |
| Ray RLlib | Distributed RL (PPO, DQN, SAC, multi-agent) |
| PyTorch | Default deep learning backend for RLModules |
| Gymnasium | Standard envs (`Taxi-v3`, `CartPole-v1`, `Pendulum-v1`) |
| `projects/*` | Self-contained demos (script, pins, README) |

Shared diagrams:

![RL overview](assets/RLOverview.png)

![RL with policy](assets/RLWithPolicy.png)

## Target Audience

- **ML engineers** exploring reinforcement learning on Cloudera AI
- **Solution architects** evaluating Ray RLlib for sequential decision use cases
- **Data scientists** comfortable with supervised ML who need a first distributed RL path
- **SEs / field engineers** delivering a short, reliable RL lab demo

## Repository Structure

| Path | Description |
| --- | --- |
| `METADATA.yaml` | Catalog metadata for the Cloudera blueprint website |
| `README.md` | Cover page — business + onboarding content |
| `assets/` | Diagrams and shared media |
| `deploy/` | Workbench sizing, multi-worker Ray, optional MLflow |
| `projects/common/` | Shared helpers (e.g. optional MLflow tracker) |
| `docs/` | Getting started, RL primer, doc index |
| `projects/` | Self-contained RLlib examples ([index](projects/README.md)) |
| `projects/taxi-ppo/` | Featured PPO → Taxi-v3 |
| `projects/cartpole-dqn/` | DQN → CartPole-v1 |
| `projects/pendulum-sac/` | SAC → Pendulum-v1 |
| `projects/multiagent-cartpole/` | Multi-agent PPO → MultiAgentCartPole |
| `projects/offline-marwil/` | MARWIL from logged CartPole episodes |
| `LICENSE` | Apache License 2.0 |

## Prerequisites

- Python **3.10–3.12**, `pip`, and `git`
- Per-project packages (same core pins): `ray[rllib]==2.56.1`, PyTorch, Gymnasium — see each `projects/*/requirements.txt`
- Cloudera AI Workbench or CML project access (for platform runs)
- Outbound PyPI (or internal mirror) for dependency install
- Optional: Jupyter / VS Code / Cursor for the Taxi notebook
- No external model API keys for these demos

## Hardware Requirements

| Deployment | Minimum |
| --- | --- |
| Cover demo (Taxi PPO) | 2 vCPU, 8 GB RAM, ~5 GB disk; GPU not required; &lt;1 min |
| CartPole DQN / multi-agent CartPole | Same as cover; roughly 1–2 minutes |
| Pendulum SAC | 2–4 vCPU, 8 GB RAM; ~5–10 minutes on CPU for the default 15-iter smoke run |
| Offline MARWIL pipeline | 2–4 vCPU, 8 GB RAM; ~3–8 minutes (record + train) |
| Larger / production-style workloads | 4+ vCPU, 16 GB RAM; GPU optional for bigger nets / image envs |

See [deploy/README.md](deploy/README.md) for Workbench session guidance.

## Documentation

- [Getting started on Cloudera AI Workbench](docs/getting-started.md)
- [RL primer](docs/rl-primer.md)
- [Deploy / sizing / multi-worker](deploy/README.md) · [MLflow tracking](deploy/mlflow.md)
- [Project index](projects/README.md)
- [Taxi PPO](projects/taxi-ppo/README.md) · [CartPole DQN](projects/cartpole-dqn/README.md) · [Pendulum SAC](projects/pendulum-sac/README.md) · [Multi-Agent CartPole](projects/multiagent-cartpole/README.md) · [Offline MARWIL](projects/offline-marwil/README.md)
- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html) · [RLlib algorithms](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html)
- [Cloudera AI documentation](https://docs.cloudera.com/)
- [Proximal Policy Optimization (paper)](https://arxiv.org/abs/1707.06347)

## Projects

| Project | Status | Algorithm | Description |
| --- | --- | --- | --- |
| [**taxi-ppo**](projects/taxi-ppo/) | Featured (cover) | PPO | Discrete logistics on Gymnasium `Taxi-v3` |
| [**cartpole-dqn**](projects/cartpole-dqn/) | Companion | DQN | Off-policy discrete control on `CartPole-v1` |
| [**pendulum-sac**](projects/pendulum-sac/) | Companion | SAC | Continuous control on `Pendulum-v1` |
| [**multiagent-cartpole**](projects/multiagent-cartpole/) | Companion | Multi-agent PPO | Fleet controllers on `MultiAgentCartPole` |
| [**offline-marwil**](projects/offline-marwil/) | Companion | MARWIL | Offline learning from logged CartPole trajectories |

Add more under `projects/<slug>/` using the [project convention](projects/README.md).

## License

Copyright 2023–2026 Ian Brooks

Licensed under the [Apache License, Version 2.0](LICENSE).
