# Deploy on Cloudera AI

This blueprint runs as code inside a **Cloudera AI Workbench** (or CML) session / notebook. It is not packaged as a container service; deployment = provision runtime → install deps → run project entrypoint.

Related: [Getting started](../docs/getting-started.md) · [MLflow tracking](mlflow.md) · [Offline BYO logs](../projects/offline-marwil/README.md#bring-your-own-parquet-logs)

## Session sizing by project (single-session smoke)

| Project | Runtime | vCPU | Memory | GPU | Typical smoke time | Entrypoint |
| --- | --- | --- | --- | --- | --- | --- |
| Taxi PPO (cover) | Python 3.10–3.12 | 2+ (4 preferred) | 8 GB | No | &lt; 1 min | `python projects/taxi-ppo/train_taxi_ppo.py` |
| CartPole DQN | Python 3.10–3.12 | 2+ | 8 GB | No | ~1–2 min | `python projects/cartpole-dqn/train_cartpole_dqn.py` |
| Pendulum SAC | Python 3.10–3.12 | 2–4 | 8 GB | No | ~5–10 min | `python projects/pendulum-sac/train_pendulum_sac.py` |
| Multi-Agent CartPole | Python 3.10–3.12 | 2+ | 8 GB | No | ~1–2 min | `python projects/multiagent-cartpole/train_multiagent_cartpole.py` |
| Offline BC / MARWIL | Python 3.10–3.12 | 2–4 | 8 GB | No | ~2–5 min | `python projects/offline-marwil/run_pipeline.py` |

Notebooks: cover `projects/taxi-ppo/RayRLTest.ipynb`; companions under each `projects/<slug>/*.ipynb` (see [project index](../projects/README.md)).

Network: PyPI (or mirror) for `pip install -r projects/<slug>/requirements.txt`.

## Multi-worker Ray on Workbench

RLlib’s main scale knob on a **single Workbench session** is `num_env_runners` (rollout workers). The session’s vCPU/memory budget must cover:

| Role | Typical CPU share | Notes |
| --- | --- | --- |
| Driver / algorithm process | ~1 vCPU | Orchestrates train loop |
| Learner(s) | ~1 vCPU (local learner) | Smoke configs use `num_learners=0` (local) |
| EnvRunners | ~1 vCPU each | `num_env_runners` remote actors |
| Evaluation EnvRunner | 0–1 vCPU | Prefer `evaluation_num_env_runners=0` when CPU-tight |
| Ray Data (offline only) | ~1 vCPU | MapBatches actors for Parquet → train batch |

### Recommended sizing

| Workbench session | Suggested `num_env_runners` | Example |
| --- | --- | --- |
| 2 vCPU / 8 GB | 1 (default smoke for most companions) | CartPole DQN |
| 4 vCPU / 8–16 GB | 2 | Cover Taxi PPO default |
| 8 vCPU / 16 GB | 4–5 | `python projects/taxi-ppo/train_taxi_ppo.py --num-env-runners 4` |
| 16 vCPU / 32 GB | 8–12 | Throughput demos / longer Taxi or CartPole runs |

**Rule of thumb:** `num_env_runners ≈ session_vCPUs − 2` (reserve driver + learner). Going higher than available CPUs causes Ray scheduling delays and “Cluster resources are not enough” warnings — especially with offline Ray Data.

### Cover demo: scale EnvRunners

```bash
# 8 vCPU session example
python projects/taxi-ppo/train_taxi_ppo.py --num-env-runners 4 --train-iters 10
```

Optional env override (same effect):

```bash
export RAY_RL_NUM_ENV_RUNNERS=4
python projects/taxi-ppo/train_taxi_ppo.py
```

### Offline project notes

- Record phase uses EnvRunners + evaluation writers — keep session ≥ 2–4 vCPU.
- Train phase should start a **fresh** Ray cluster (`run_pipeline.py` calls `ray.shutdown()` between phases).
- BYO Parquet (skip recorder): see [offline-marwil README](../projects/offline-marwil/README.md#bring-your-own-parquet-logs).

### Multi-node / Ray cluster

These labs target **one Workbench session** (local Ray). Multi-node Ray on Cloudera AI is possible for larger jobs but is out of scope for the smoke ladder — keep hyperparams and worker counts inside each project’s script until you promote a workload to a dedicated Ray cluster pattern.

## Experiment tracking (optional MLflow)

Log `episode_return_mean` (and eval return) to MLflow — useful when Workbench or an external tracking server is configured:

```bash
pip install mlflow
export MLFLOW_TRACKING_URI=./mlruns   # or your Workbench / remote URI
export RAY_RL_MLFLOW=1
python projects/taxi-ppo/train_taxi_ppo.py
```

Details: [mlflow.md](mlflow.md).

## Production / larger RL workloads

| Setting | Guidance |
| --- | --- |
| vCPU | 4–16+ depending on `num_env_runners` |
| Memory | 16 GB+ |
| GPU | Optional; enable when switching to larger nets / image envs |
| Storage | Space for `ray_results/`, checkpoints, offline Parquet, and `mlruns/` |
| Scaling | Raise EnvRunners first; then consider multi-node Ray |

Exact hyperparams stay in each project’s script and README.

## Security / access

- Cloudera AI project membership and session launch permission
- No external API keys required for these Gymnasium demos
- MLflow remote tracking may need network allowlisting to your tracking server
- Outbound access only as needed for package install (or use an air-gapped wheelhouse)
