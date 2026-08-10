# Offline RL / BC (from logged CartPole trajectories)

Learn a policy **from logged experience** — no online exploration during the offline phase.

This is the blueprint’s offline / “ops logs” lab: when rolling out a live policy is costly or risky, train from trajectories you already have.

| | |
| --- | --- |
| Record logs | `record_cartpole_logs.py` |
| Train offline | `train_offline_marwil.py` |
| One-shot smoke | `run_pipeline.py` |
| Notebook | [`offline_bc.ipynb`](offline_bc.ipynb) |
| Smoke algorithm | Behavior Cloning (`BCConfig`) |
| Optional upgrade | MARWIL (`MARWILConfig`, `beta>0`) |
| Env (spaces + eval) | `CartPole-v1` |
| Data format | RLlib episode Parquet under `data/cartpole/` |

## The problem

In many enterprises you have **historical trajectories** (dispatcher logs, robot teleop, past controller rollouts) but limited ability to keep exploring online. Offline RL / imitation answers:

> Given logged `(observation, action, reward, …)` sequences, learn a policy that performs well when evaluated in the simulator — without collecting new exploratory actions during training.

This project:

1. Trains a short CartPole PPO **behavior** policy (stand-in for an existing controller)
2. **Records** evaluation episodes to disk
3. Trains **BC** only from those files (MARWIL-ready wiring)
4. **Evaluates** the learned policy in CartPole

## Why BC here (and when to use MARWIL)

| Approach | Fit |
| --- | --- |
| Online PPO/DQN/SAC | Needs continued env interaction; may be unsafe or expensive |
| **BC** (this smoke) | Pure imitation of logged actions — fast, stable on short demos |
| **MARWIL** (`beta>0`) | Imitation **plus** advantage re-weighting — better when logs are mixed-quality and you have more data |

Swap `BCConfig` → `MARWILConfig` and set `beta=1.0` (usually with lower `lr` and more data) for full advantage-weighted MARWIL. The `.offline_data(input_=...)` block stays the same ([RLlib offline docs](https://docs.ray.io/en/latest/rllib/rllib-offline.html)).

Enterprise analogy: improve a policy from warehouse / plant / ticket-routing logs before enabling live exploration.

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r projects/offline-marwil/requirements.txt

# Record + train in one go (~2–4 min)
python projects/offline-marwil/run_pipeline.py
```

Notebook twin: [`offline_bc.ipynb`](offline_bc.ipynb) (record → Ray shutdown → train cells).

Or step by step:

```bash
python projects/offline-marwil/record_cartpole_logs.py
python projects/offline-marwil/train_offline_marwil.py
```

Generated data lives under `projects/offline-marwil/data/` (gitignored). Re-run record to refresh logs.

## Bring your own Parquet logs

Skip the PPO recorder when you already have RLlib-compatible offline files (for example, exports sitting on a Workbench mount or object store synced locally):

```bash
# Episode Parquet produced by RLlib offline_data(output=..., output_write_episodes=True)
python projects/offline-marwil/train_offline_marwil.py \
  --input /path/to/your/episode_parquet_dir

# Same via env (handy in Workbench jobs):
export RAY_RL_OFFLINE_INPUT=/path/to/your/episode_parquet_dir
python projects/offline-marwil/train_offline_marwil.py
```

| Flag / env | Meaning |
| --- | --- |
| `--input PATH` | Directory (or file) of Parquet logs |
| `RAY_RL_OFFLINE_INPUT` | Default for `--input` when the flag is omitted |
| `--timesteps` | Columnar timestep rows instead of episode objects (`input_read_episodes=False`) |

**Expected format for this smoke:** RLlib episode Parquet as written by `record_cartpole_logs.py` (nested under something like `…/cartpoleenv/run-…/*.parquet`, schema typically a binary `item` column). Spaces must match `CartPole-v1` for evaluation. For other envs, change `.environment(...)` in `train_offline_marwil.py` to match your data.

Optional MLflow: see [deploy/mlflow.md](../../deploy/mlflow.md).

## What success looks like

- **Record phase:** PPO returns climb; evaluation rollouts write Parquet under `data/cartpole/` (behavior eval often ~200–400).
- **Offline phase:** Each iter prints `evaluate_episode_return_mean`. Expect a climb from ~50–120 toward **~180–280** over ~20 iters (variance is normal late in the run).

Ignore Ray “new API stack” / `RLModule` deprecation warnings. If you previously saw `Cluster resources are not enough` during a back-to-back pipeline run, `run_pipeline.py` now shuts Ray down between record and train.

## Architecture

```text
[Online, short] PPO on CartPole ──save──► behavior checkpoint
        │
        ▼ evaluate() + offline_data(output=...)
Logged episode Parquet  (data/cartpole/)
        │
        ▼ offline_data(input_=..., input_read_episodes=True)
BC learner (no env sampling for updates)
        │
        ▼ evaluation EnvRunner
Online CartPole score of the offline-trained policy
```

## Requirements

- Python 3.10–3.12
- [`requirements.txt`](requirements.txt): `ray[rllib]`, Torch, Gymnasium, PyArrow, msgpack-numpy

## See also

- [CartPole DQN](../cartpole-dqn/) — same env, online off-policy
- [Taxi PPO](../taxi-ppo/) — cover online on-policy demo
- [RLlib offline RL](https://docs.ray.io/en/latest/rllib/rllib-offline.html)
- [MARWIL in algorithms](https://docs.ray.io/en/latest/rllib/rllib-algorithms.html#marwil)
