# Taxi PPO

Train **Proximal Policy Optimization (PPO)** on Gymnasium [`Taxi-v3`](https://gymnasium.farama.org/environments/toy_text/taxi/) with the current Ray RLlib API stack.

This is the **cover demo** for the [Ray RLlib on Cloudera AI](../../README.md) blueprint.

| | |
| --- | --- |
| Entrypoint | `train_taxi_ppo.py` |
| Notebook | `RayRLTest.ipynb` |
| Workbench guide | [docs/getting-started.md](../../docs/getting-started.md) |
| Catalog metadata | [`METADATA.yaml`](../../METADATA.yaml) (`featured_project`) |

## Quickstart

From the repository root:

```bash
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r projects/taxi-ppo/requirements.txt

python projects/taxi-ppo/train_taxi_ppo.py
```

Or from this directory:

```bash
cd projects/taxi-ppo
pip install -r requirements.txt
python train_taxi_ppo.py
```

Notebook twin: [`RayRLTest.ipynb`](RayRLTest.ipynb) (select the same venv kernel).

## What it does

1. Configure PPO on `Taxi-v3` with two EnvRunners
2. Flatten discrete Taxi observations with `FlattenObservations`
3. Train for five iterations, evaluate, then `algo.stop()`

## Expected output

Mean episode return should improve across iterations (still negative for a short Taxi run), then a final evaluate line, for example:

```text
iter=1  episode_return_mean=-747.9  env_steps=4000.0
…
iter=5  episode_return_mean=-410.6  env_steps=20000.0
evaluate  episode_return_mean=-353.9
```

Ray may still print internal deprecation notices; they are harmless for this smoke test.

## Architecture

```text
PPO Algorithm + Torch RLModule
        │ sample / sync
        ▼
EnvRunners (×2) ── Taxi-v3 ── FlattenObservations
        │
        ▼
Eval EnvRunner (manual algo.evaluate())
```

## Requirements

- Python 3.10–3.12
- See [`requirements.txt`](requirements.txt) (`ray[rllib]==2.56.1`, Torch, Gymnasium)

Apple Silicon: [Ray M-series install notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support).
