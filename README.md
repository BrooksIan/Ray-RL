# Cloudera Blueprint: Ray RLlib on Cloudera AI

> Multi-project **Ray RLlib** examples for Cloudera AI Workbench. The cover demo trains **PPO on Gymnasium Taxi-v3**. Catalog fields: [`METADATA.yaml`](METADATA.yaml).

## Table of Contents

- [Overview](#overview)
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

**Ray RLlib on Cloudera AI** helps ML engineers and architects learn distributed reinforcement learning where they already work — inside Cloudera AI Workbench. The blueprint ships a featured PPO → Taxi-v3 quickstart (script + notebook) on RLlib’s current EnvRunner / RLModule API, plus a `projects/` layout for adding more environments without rewriting the repo. Cloudera value: run Ray-based RL experiments on the same governed AI workbench used for notebooks, sessions, and team collaboration.

## Demo

Primary demo (no Reprise recording required):

1. Follow [Quickstart](#quickstart) or [Workbench getting started](docs/getting-started.md).
2. Run:

   ```bash
   python projects/taxi-ppo/train_taxi_ppo.py
   ```

3. Confirm episode return improves across five iterations, then an `evaluate` line prints.

Interactive twin: [`projects/taxi-ppo/RayRLTest.ipynb`](projects/taxi-ppo/RayRLTest.ipynb).

Sample output and troubleshooting: [`projects/taxi-ppo/README.md`](projects/taxi-ppo/README.md).

Optional catalog field: set `reprise_link` in [`METADATA.yaml`](METADATA.yaml) when a recorded walkthrough is published.

## Use Case

**Problem:** Teams new to RL struggle to connect vocabulary (observation, reward, policy) to a modern distributed stack, and many tutorials use outdated RLlib APIs.

**Outcome:** A reproducible first success on Cloudera AI — train and evaluate PPO on Taxi-v3 in minutes — with a clear path to grow a library of RL projects (new envs, algorithms, multi-agent) under one blueprint.

Relevant for control-style decisioning exploration: simulation, operations research prototypes, robotics labs, and sequential decision POCs before production hardening.

## Key Features

- Featured **Taxi PPO** cover demo on the current RLlib API stack
- **Multi-project** layout (`projects/<slug>/`) for expanding beyond Taxi
- Runnable **script + notebook** with pinned dependencies
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

**Cloudera AI Workbench:** start a Python session (≥2 vCPU / 8 GB), clone or open this repo, then run the same install + entrypoint steps. Full guide: [docs/getting-started.md](docs/getting-started.md).

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).

## Software Components

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
| Ray RLlib | Distributed RL algorithms and training loop |
| PyTorch | Default deep learning backend for RLModules |
| Gymnasium | Standard RL environments (`Taxi-v3` cover demo) |
| `projects/taxi-ppo` | Featured entrypoint, notebook, and pins |

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
| `deploy/` | Workbench / session deployment notes |
| `docs/` | Getting started, RL primer, doc index |
| `projects/` | Self-contained RLlib examples |
| `projects/taxi-ppo/` | Featured PPO → Taxi-v3 project |
| `LICENSE` | Apache License 2.0 |

## Prerequisites

- Python **3.10–3.12**, `pip`, and `git`
- Cover demo packages: [`projects/taxi-ppo/requirements.txt`](projects/taxi-ppo/requirements.txt) (`ray[rllib]==2.56.1`, PyTorch, Gymnasium)
- Cloudera AI Workbench or CML project access (for platform runs)
- Outbound PyPI (or internal mirror) for dependency install
- Optional: Jupyter / VS Code / Cursor for the notebook
- No external model API keys for the Taxi cover demo

## Hardware Requirements

| Deployment | Minimum |
| --- | --- |
| Launchable / demo (Taxi PPO) | 2 vCPU, 8 GB RAM, ~5 GB disk for deps; GPU not required |
| Production / larger envs | 4+ vCPU, 16 GB RAM; GPU optional for larger nets / image envs |

The Taxi-v3 smoke test typically finishes in under a minute on a laptop or small Workbench session. See [deploy/README.md](deploy/README.md) for scaling notes.

## Documentation

- [Getting started on Cloudera AI Workbench](docs/getting-started.md)
- [RL primer](docs/rl-primer.md)
- [Deploy / sizing](deploy/README.md)
- [Taxi PPO project README](projects/taxi-ppo/README.md)
- [How to add a project](projects/README.md)
- [RLlib docs](https://docs.ray.io/en/latest/rllib/index.html)
- [Cloudera AI documentation](https://docs.cloudera.com/)
- [Proximal Policy Optimization (paper)](https://arxiv.org/abs/1707.06347)

## Projects

| Project | Status | Description |
| --- | --- | --- |
| [**taxi-ppo**](projects/taxi-ppo/) | Featured (cover) | Modern RLlib PPO on Gymnasium `Taxi-v3` |

Add more under `projects/<slug>/` using the [project convention](projects/README.md).

## License

Copyright 2023–2026 Ian Brooks

Licensed under the [Apache License, Version 2.0](LICENSE).
