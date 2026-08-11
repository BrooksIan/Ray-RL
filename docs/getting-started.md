# Getting started on Cloudera AI Workbench

Run the featured **Taxi PPO** cover demo, then optionally continue the project ladder, in a Cloudera AI Workbench (or CML) session.

## 1. Open a session

1. In your Cloudera AI project, start a **Python** session or notebook runtime.
2. Recommended session size for the cover demo: **2+ vCPU**, **8 GB memory**, no GPU required.
3. For **Pendulum SAC**, prefer **4 vCPU** if available (default smoke run is longer).
4. Ensure the runtime can install packages from PyPI (or your internal mirror).

## 2. Get the code

```bash
git clone https://github.com/BrooksIan/Ray-RL.git
cd Ray-RL
```

If the repo is already mounted as the project filesystem, `cd` to its root instead.

## 3. Install dependencies (cover demo)

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r projects/taxi-ppo/requirements.txt
```

Companion projects use the same core pins; install from each project’s `requirements.txt` before running that entrypoint.

## 4. Run the cover demo

```bash
python projects/taxi-ppo/train_taxi_ppo.py
```

Or open `projects/taxi-ppo/RayRLTest.ipynb` and run all cells with the same environment.

Each companion project also has a notebook twin (same deps as its `requirements.txt`):

| Project | Notebook |
| --- | --- |
| CartPole DQN | `projects/cartpole-dqn/cartpole_dqn.ipynb` |
| Pendulum SAC | `projects/pendulum-sac/pendulum_sac.ipynb` |
| Multi-Agent CartPole | `projects/multiagent-cartpole/multiagent_cartpole.ipynb` |
| Offline BC | `projects/offline-marwil/offline_bc.ipynb` |
| Custom playground PPO | `projects/custom-env-ppo/custom_env_ppo.ipynb` |

## 5. What success looks like (Taxi)

Within about a minute you should see five training iterations with improving `episode_return_mean`, then an `evaluate` line. Details: [projects/taxi-ppo/README.md](../projects/taxi-ppo/README.md).

## 6. Continue the learning path

| Project | Install | Run | Typical smoke time |
| --- | --- | --- | --- |
| [CartPole DQN](../projects/cartpole-dqn/README.md) | `pip install -r projects/cartpole-dqn/requirements.txt` | `python projects/cartpole-dqn/train_cartpole_dqn.py` | ~1–2 min |
| [Pendulum SAC](../projects/pendulum-sac/README.md) | `pip install -r projects/pendulum-sac/requirements.txt` | `python projects/pendulum-sac/train_pendulum_sac.py` | ~5–10 min |
| [Multi-Agent CartPole](../projects/multiagent-cartpole/README.md) | `pip install -r projects/multiagent-cartpole/requirements.txt` | `python projects/multiagent-cartpole/train_multiagent_cartpole.py` | ~1–2 min |
| [Offline BC / MARWIL](../projects/offline-marwil/README.md) | `pip install -r projects/offline-marwil/requirements.txt` | `python projects/offline-marwil/run_pipeline.py` | ~2–5 min |
| [Custom playground PPO](../projects/custom-env-ppo/README.md) | `pip install -r projects/custom-env-ppo/requirements.txt` | `python projects/custom-env-ppo/train_queue_ppo.py` | ~1–2 min |

Full ladder description: [README learning path](../README.md#learning-path).

## 7. Platform options (Workbench)

| Goal | How |
| --- | --- |
| More rollout throughput | Larger session + `python projects/taxi-ppo/train_taxi_ppo.py --num-env-runners N` — see [deploy multi-worker](../deploy/README.md#multi-worker-ray-on-workbench) |
| Track episode return in MLflow | `pip install mlflow` then `RAY_RL_MLFLOW=1` (default store `sqlite:///./mlflow.db`) — [deploy/mlflow.md](../deploy/mlflow.md) |
| Offline from existing logs | Skip the recorder: `train_offline_marwil.py --input /path/to/parquet` — [BYO logs](../projects/offline-marwil/README.md#bring-your-own-parquet-logs) |
| Custom business process env | Edit + train `TicketQueue-v0` — [custom-env-ppo](../projects/custom-env-ppo/README.md) |
| Offline from custom playground | `python projects/custom-env-ppo/run_offline_pipeline.py` — [offline loop](../projects/custom-env-ppo/README.md#offline-loop-logs-from-your-playground) |

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| Ray / Torch install fails on Apple Silicon | Follow [Ray M-series notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support) |
| `PermissionError` under `~/ray_results` | Ensure the session home is writable, or set `HOME` to a writable project path |
| Out of memory during install | Use a larger session or install CPU Torch wheels only |
| Old RLlib API errors | Confirm `ray[rllib]==2.56.1` from the project `requirements.txt` |
| `python train_*.py` not found at repo root | Entrypoints live under `projects/<slug>/` — use the paths above |
| Pendulum returns stay very negative | Expected early on; look for the mean becoming *less* negative over iters |
| `Cluster resources are not enough` | Lower `num_env_runners`, or use a larger session; for offline, prefer `run_pipeline.py` (shuts Ray down between record and train) |

## Next projects

Add sibling folders under [`projects/`](../projects/README.md). Keep each project self-contained with its own `README.md` and `requirements.txt`.
