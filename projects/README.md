# Projects

Each subdirectory under `projects/` is a self-contained Ray RLlib example.

## Learning path

| Step | Project | Path | Algorithm | Notebook | Summary |
| --- | --- | --- | --- | --- | --- |
| 1 | **Taxi PPO** (cover) | [`taxi-ppo/`](taxi-ppo/) | PPO | [`RayRLTest.ipynb`](taxi-ppo/RayRLTest.ipynb) | Discrete logistics on `Taxi-v3` |
| 2 | **CartPole DQN** | [`cartpole-dqn/`](cartpole-dqn/) | DQN | [`cartpole_dqn.ipynb`](cartpole-dqn/cartpole_dqn.ipynb) | Off-policy discrete control on `CartPole-v1` |
| 3 | **Pendulum SAC** | [`pendulum-sac/`](pendulum-sac/) | SAC | [`pendulum_sac.ipynb`](pendulum-sac/pendulum_sac.ipynb) | Continuous control on `Pendulum-v1` |
| 4 | **Multi-Agent CartPole PPO** | [`multiagent-cartpole/`](multiagent-cartpole/) | Multi-agent PPO | [`multiagent_cartpole.ipynb`](multiagent-cartpole/multiagent_cartpole.ipynb) | Fleet controllers on `MultiAgentCartPole` |
| 5 | **Offline BC / MARWIL** | [`offline-marwil/`](offline-marwil/) | BC (MARWIL-ready) | [`offline_bc.ipynb`](offline-marwil/offline_bc.ipynb) | Learn from logged CartPole trajectories |

Cover quickstart stays on Taxi; companions are optional next labs. Blueprint overview: [../README.md](../README.md).

## Layout convention

```text
projects/<project-slug>/
  README.md           # purpose, why RL, quickstart, expected output
  requirements.txt    # pinned deps for this project
  train_*.py          # runnable entrypoint
  *.ipynb             # optional notebook twin
```

Optional later: `assets/`, `configs/`, `tests/` inside a project as needed.

Shared helpers (optional MLflow, etc.) live under [`common/`](common/). Platform sizing and tracking: [`../deploy/README.md`](../deploy/README.md).

## Add a new project

1. Copy an existing project (start from [`taxi-ppo/`](taxi-ppo/) or [`cartpole-dqn/`](cartpole-dqn/)) or create a new folder:

   ```bash
   mkdir -p projects/my-env-algo
   ```

2. Add `README.md`, `requirements.txt`, and at least one runnable script.
3. Link it from this file and from the root [`README.md`](../README.md) **Learning path** and **Projects** tables.
4. Update [`docs/getting-started.md`](../docs/getting-started.md) and [`deploy/README.md`](../deploy/README.md) if session sizing differs.
5. Keep the cover README’s **Demo / Quickstart** focused on `taxi-ppo` unless you intentionally change the cover story.
