# Deploy on Cloudera AI

This blueprint runs as code inside a **Cloudera AI Workbench** (or CML) session / notebook. It is not packaged as a container service; deployment = provision runtime → install deps → run project entrypoint.

## Cover demo (Taxi PPO)

| Setting | Recommendation |
| --- | --- |
| Runtime | Python 3.10–3.12 |
| vCPU | 2 minimum (4 preferred) |
| Memory | 8 GB minimum |
| GPU | Not required for Taxi-v3 smoke test |
| Network | PyPI (or mirror) for `pip install` |
| Entrypoint | `python projects/taxi-ppo/train_taxi_ppo.py` |
| Notebook | `projects/taxi-ppo/RayRLTest.ipynb` |

Step-by-step: [docs/getting-started.md](../docs/getting-started.md).

## Production / larger RL workloads

| Setting | Guidance |
| --- | --- |
| vCPU | 4–16+ depending on `num_env_runners` |
| Memory | 16 GB+ |
| GPU | Optional; enable when switching to larger nets / image envs |
| Storage | Space for `ray_results/`, checkpoints, and env assets |
| Scaling | Increase EnvRunners / learners in project configs; consider Ray cluster patterns for multi-node |

Exact sizing belongs in each project's README as new examples are added.

## Security / access

- Cloudera AI project membership and session launch permission
- No external API keys required for the Taxi PPO cover demo
- Outbound access only as needed for package install (or use an air-gapped wheelhouse)
