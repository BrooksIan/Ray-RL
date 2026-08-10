# Getting started on Cloudera AI Workbench

This guide runs the featured **Taxi PPO** project in a Cloudera AI Workbench (or CML) session.

## 1. Open a session

1. In your Cloudera AI project, start a **Python** session or notebook runtime.
2. Recommended session size for the cover demo: **2+ vCPU**, **8 GB memory**, no GPU required.
3. Ensure the runtime can install packages from PyPI (or your internal mirror).

## 2. Get the code

```bash
git clone https://github.com/BrooksIan/Ray-RL.git
cd Ray-RL
```

If the repo is already mounted as the project filesystem, `cd` to its root instead.

## 3. Install dependencies

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r projects/taxi-ppo/requirements.txt
```

## 4. Run the cover demo

```bash
python projects/taxi-ppo/train_taxi_ppo.py
```

Or open `projects/taxi-ppo/RayRLTest.ipynb` and run all cells with the same environment.

## 5. What success looks like

Within about a minute you should see five training iterations with improving `episode_return_mean`, then an `evaluate` line. Details and sample output: [projects/taxi-ppo/README.md](../projects/taxi-ppo/README.md).

## Troubleshooting

| Symptom | What to try |
| --- | --- |
| Ray / Torch install fails on Apple Silicon | Follow [Ray M-series notes](https://docs.ray.io/en/latest/ray-overview/installation.html#m1-mac-apple-silicon-support) |
| `PermissionError` under `~/ray_results` | Ensure the session home is writable, or set `HOME` to a writable project path |
| Out of memory during install | Use a larger session or install CPU Torch wheels only |
| Old RLlib API errors | Confirm `ray[rllib]==2.56.1` from this project's `requirements.txt` |

## Next projects

Add sibling folders under [`projects/`](../projects/README.md) (CartPole, custom envs, multi-agent). Keep each project self-contained with its own `README.md` and `requirements.txt`.
