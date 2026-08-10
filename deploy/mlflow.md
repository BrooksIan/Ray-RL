# Optional MLflow experiment tracking

Log RLlib **episode return** metrics to [MLflow](https://mlflow.org/) so Cloudera AI Workbench runs show up alongside other experiments.

Tracking is **opt-in**. Default project scripts print to stdout only and do not require MLflow.

## Enable

```bash
pip install mlflow
# Local SQLite store (default if unset when RAY_RL_MLFLOW=1):
export MLFLOW_TRACKING_URI="${MLFLOW_TRACKING_URI:-sqlite:///./mlflow.db}"
export RAY_RL_MLFLOW=1

python projects/taxi-ppo/train_taxi_ppo.py
# or
python projects/offline-marwil/train_offline_marwil.py --input projects/offline-marwil/data/cartpole
```

| Variable | Effect |
| --- | --- |
| `RAY_RL_MLFLOW=1` | Enable logging via [`projects/common/rllib_mlflow.py`](../projects/common/rllib_mlflow.py) |
| `MLFLOW_TRACKING_URI` | SQLite URI, HTTP tracking server, or legacy file store |
| `RAY_RL_MLFLOW_EXPERIMENT` | Experiment name (default: `ray-rllib-blueprint`) |

View local runs (leave this process running, then open the URL it prints — usually `http://127.0.0.1:5000`):

```bash
mlflow ui --backend-store-uri "${MLFLOW_TRACKING_URI:-sqlite:///./mlflow.db}"
```

Look for experiment **`ray-rllib-blueprint`** and runs such as `taxi-ppo` or `offline-bc`. Metrics include `episode_return_mean` / `evaluate_episode_return_mean` by training step.

### Legacy `./mlruns` file store

Recent MLflow versions put the filesystem backend in maintenance mode. Prefer SQLite (above). If you must keep `./mlruns`:

```bash
export MLFLOW_TRACKING_URI=./mlruns
export MLFLOW_ALLOW_FILE_STORE=true
export RAY_RL_MLFLOW=1
```

The helper also sets `MLFLOW_ALLOW_FILE_STORE=true` automatically when the URI looks like a classic `mlruns` path.

## Metrics logged

| Metric | When |
| --- | --- |
| `episode_return_mean` | Each training iteration (online projects) |
| `evaluate_episode_return_mean` | Offline eval return, or cover demo final `evaluate()` |
| `env_steps` | When present in the RLlib result dict |

Also logs simple params: project slug, algorithm hint, `num_env_runners` when known.

## Workbench tips

- Prefer a **project-local** `MLFLOW_TRACKING_URI` under the Workbench filesystem so runs survive session restart.
- If your site hosts a central MLflow server, point `MLFLOW_TRACKING_URI` at that HTTP endpoint and ensure the session can reach it.
- Do not commit `mlflow.db` / `mlruns/` — both are gitignored at the repo root.

## Wiring in other projects

```python
from projects.common.rllib_mlflow import EpisodeReturnTracker

tracker = EpisodeReturnTracker(run_name="cartpole-dqn", tags={"project": "cartpole-dqn"})
# inside train loop:
tracker.log_train(iteration=i, episode_return_mean=ret, env_steps=steps)
tracker.close()
```

Import works when you run from the **repository root** (scripts add the root to `sys.path` as needed).
