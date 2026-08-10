# Deploy on Cloudera AI

This blueprint runs as code inside a **Cloudera AI Workbench** (or CML) session / notebook. It is not packaged as a container service; deployment = provision runtime → install deps → run project entrypoint.

## Session sizing by project

| Project | Runtime | vCPU | Memory | GPU | Typical smoke time | Entrypoint |
| --- | --- | --- | --- | --- | --- | --- |
| Taxi PPO (cover) | Python 3.10–3.12 | 2+ (4 preferred) | 8 GB | No | &lt; 1 min | `python projects/taxi-ppo/train_taxi_ppo.py` |
| CartPole DQN | Python 3.10–3.12 | 2+ | 8 GB | No | ~1–2 min | `python projects/cartpole-dqn/train_cartpole_dqn.py` |
| Pendulum SAC | Python 3.10–3.12 | 2–4 | 8 GB | No | ~5–10 min | `python projects/pendulum-sac/train_pendulum_sac.py` |
| Multi-Agent CartPole | Python 3.10–3.12 | 2+ | 8 GB | No | ~1–2 min | `python projects/multiagent-cartpole/train_multiagent_cartpole.py` |

Cover notebook: `projects/taxi-ppo/RayRLTest.ipynb`.

Network: PyPI (or mirror) for `pip install -r projects/<slug>/requirements.txt`.

Step-by-step: [docs/getting-started.md](../docs/getting-started.md).

## Production / larger RL workloads

| Setting | Guidance |
| --- | --- |
| vCPU | 4–16+ depending on `num_env_runners` |
| Memory | 16 GB+ |
| GPU | Optional; enable when switching to larger nets / image envs |
| Storage | Space for `ray_results/`, checkpoints, and env assets |
| Scaling | Increase EnvRunners / learners in project configs; consider Ray cluster patterns for multi-node |

Exact hyperparams stay in each project’s script and README.

## Security / access

- Cloudera AI project membership and session launch permission
- No external API keys required for these Gymnasium demos
- Outbound access only as needed for package install (or use an air-gapped wheelhouse)
