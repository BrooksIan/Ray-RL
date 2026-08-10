# Cloudera Blueprint: Ray RLlib on Cloudera AI

> A multi-project collection of **Ray RLlib** examples for Cloudera AI Workbench. The cover demo is **PPO on Gymnasium Taxi-v3**. Catalog fields live in [`METADATA.yaml`](METADATA.yaml).

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
- [Demo](#demo)
- [Use Case](#use-case)
- [Key Features](#key-features)
- [Quickstart](#quickstart)
- [Architecture](#architecture)
- [Target Audience](#target-audience)
- [Repository Structure](#repository-structure)
- [Prerequisites](#prerequisites)
- [Hardware Requirements](#hardware-requirements)
- [Documentation](#documentation)

## Overview

This blueprint teaches distributed reinforcement learning with **Ray RLlib** and hosts multiple self-contained projects under [`projects/`](projects/). The featured path trains PPO on Taxi-v3 with the current EnvRunner / RLModule API so teams can go from RL vocabulary to a first successful `algo.train()` / `algo.evaluate()` run in minutes — then grow into additional environments and recipes in sibling project folders.

## Projects

| Project | Status | Description |
| --- | --- | --- |
| [**taxi-ppo**](projects/taxi-ppo/) | Featured (cover) | Modern RLlib PPO on Gymnasium `Taxi-v3` |
| *[your next project]* | — | Add under `projects/<slug>/` — see [projects/README.md](projects/README.md) |

## Demo

**Cover demo — Taxi PPO**

```bash
pip install -r projects/taxi-ppo/requirements.txt
python projects/taxi-ppo/train_taxi_ppo.py
```

Or open [`projects/taxi-ppo/RayRLTest.ipynb`](projects/taxi-ppo/RayRLTest.ipynb).

Expected signal after five short iterations: mean episode return improves (still negative on Taxi-v3 for a tiny run), then a final `evaluate` line prints. Full notes: [`projects/taxi-ppo/README.md`](projects/taxi-ppo/README.md).

A recorded Reprise walkthrough can be linked when available (`reprise_link` in [`METADATA.yaml`](METADATA.yaml)).

## Use Case

Teams adopting RL often struggle to connect textbook terms (observation, reward, policy) to a modern, distributed training stack. This blueprint closes that gap with a catalog of focused projects: start with **PPO on Taxi**, then add CartPole, custom envs, or multi-agent recipes without rewriting the repo. Primary outcome — a reproducible first success path for RL on Cloudera AI that scales as a project library.

## Key Features

- Multi-project layout under [`projects/`](projects/) with a clear add-a-project convention
- Featured Taxi PPO demo using the modern RLlib API (`env_runners`, connectors, RLModules)
- Per-project `requirements.txt` and README for isolated experiments
- Concise RL terminology and diagrams for onboarding non-RL practitioners
- Catalog-ready [`METADATA.yaml`](METADATA.yaml) for Cloudera blueprint listing

## Quickstart

1. Clone the repository.
2. Create and activate a virtual environment (Python **3.10–3.12**):

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # Windows: .venv\Scripts\activate
   ```

3. Run the cover project (Taxi PPO):

   ```bash
   pip install -r projects/taxi-ppo/requirements.txt
   python projects/taxi-ppo/train_taxi_ppo.py
   ```

4. Or open [`projects/taxi-ppo/RayRLTest.ipynb`](projects/taxi-ppo/RayRLTest.ipynb) with the venv kernel selected.

**On Cloudera AI Workbench / CML:** create a Python session or notebook runtime, clone this repo, install the project requirements, then run the script or notebook.

Apple Silicon: see [Ray install notes for M-series Macs](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support) if the default wheel fails.

## Architecture

Cover demo (Taxi PPO):

```text
Developer / Workbench session
        │
        ▼
┌───────────────────┐     sample      ┌────────────────────┐
│  PPO Algorithm    │◄───────────────►│ EnvRunners (×2)    │
│  + RLModule MLP   │                 │ Gymnasium Taxi-v3  │
│  (Torch)          │                 │ FlattenObservations│
└─────────┬─────────┘                 └────────────────────┘
          │ evaluate()
          ▼
┌───────────────────┐
│ Eval EnvRunner    │
└───────────────────┘
```

Repo shape:

```text
Ray-RL/                     # blueprint cover + catalog metadata
  images/                   # shared RL diagrams
  projects/
    taxi-ppo/               # featured demo
    <next-project>/         # additional examples
```

| Component | Role |
| --- | --- |
| Ray RLlib PPO | On-policy trainer (`PPOConfig` → `build_algo()`) |
| EnvRunners | Parallel environment sampling |
| `FlattenObservations` | One-hot discrete Taxi observations for the MLP |
| Per-project folder | Isolated deps, script, and notebook |
| Cloudera AI Workbench | Host for notebook/session execution |

RL loop reminder:

![RL overview](images/RLOverview.png)

![RL with policy](images/RLWithPolicy.png)

## Target Audience

- ML engineers exploring reinforcement learning on Cloudera AI
- Solution architects evaluating Ray RLlib for control / sequential decision use cases
- Data scientists who know supervised ML and need a first distributed RL path

## Repository Structure

| Path | Description |
| --- | --- |
| `METADATA.yaml` | Catalog metadata for the Cloudera blueprint website |
| `README.md` | Cover page — overview, featured demo, multi-project index |
| `projects/` | Self-contained RLlib examples ([convention](projects/README.md)) |
| `projects/taxi-ppo/` | Featured PPO → Taxi-v3 project |
| `images/` | Shared diagrams (RL overview and policy) |
| `LICENSE` | Apache License 2.0 |

## Prerequisites

- Python **3.10–3.12**
- `pip` and git
- Project deps (cover): [`projects/taxi-ppo/requirements.txt`](projects/taxi-ppo/requirements.txt)
- Optional: Cloudera AI Workbench / CML session with outbound package install access
- Optional: Jupyter / VS Code / Cursor for notebooks

## Hardware Requirements

| Deployment | Minimum |
| --- | --- |
| Local / Workbench demo (Taxi) | 2+ CPU cores, 8 GB RAM, ~5 GB disk for deps |
| Longer training / larger envs | 4+ CPUs, 16 GB RAM; GPU optional (Torch CUDA) |

The Taxi-v3 smoke test completes in under a minute on a laptop CPU. Other projects may list their own sizing in their README.

## Documentation

### RL terminology (primer)

An RL environment consists of:

1. **Action space** — all possible actions
2. **State space** — a complete description of the environment (nothing hidden)
3. **Observation space** — what the agent actually sees from the state
4. **Reward** — the only feedback the agent receives after each action

The model that tries to maximize the expected sum of future rewards is a **policy**. It maps observations to actions, usually written π(s(t)) → a(t). In deep RL, that function is a neural network.

> In RL, “model” is roughly equivalent to “policy,” but policy is more specific because it is trained in a particular environment. For deployment, people often say “model” in the usual ML sense.

### External links

- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html)
- [RLlib examples](https://docs.ray.io/en/latest/rllib/rllib-examples.html)
- [Intro to RLlib: example environments](https://medium.com/distributed-computing-with-ray/intro-to-rllib-example-environments-3a113f532c70)
- [Proximal Policy Optimization (paper)](https://arxiv.org/abs/1707.06347)
- [Deep RL: Pong from pixels](http://karpathy.github.io/2016/05/31/rl/)
- [Cloudera AI documentation](https://docs.cloudera.com/)

## License

Copyright 2023–2026 Ian Brooks

Licensed under the [Apache License, Version 2.0](LICENSE).
