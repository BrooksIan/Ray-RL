"""Train PPO on Gymnasium Taxi-v3 with Ray RLlib (new API stack)."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

# Ray 2.56 still emits this from inside default RLModule construction.
warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*",
    category=DeprecationWarning,
)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.connectors.env_to_module import FlattenObservations
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

# Repo root on path for optional projects.common helpers.
_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.common.rllib_mlflow import EpisodeReturnTracker


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    return float(value) if value is not None else None


def _parse_args() -> argparse.Namespace:
    env_runners_default = int(os.environ.get("RAY_RL_NUM_ENV_RUNNERS", "2"))
    parser = argparse.ArgumentParser(description="PPO on Taxi-v3 (cover demo)")
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=env_runners_default,
        help="RLlib EnvRunner workers (scale with Workbench vCPUs; default 2 or RAY_RL_NUM_ENV_RUNNERS)",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=5,
        help="Number of training iterations (default: 5)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_env_runners < 0:
        raise SystemExit("--num-env-runners must be >= 0")

    tracker = EpisodeReturnTracker(
        run_name="taxi-ppo",
        tags={"project": "taxi-ppo", "algorithm": "PPO"},
        params={
            "env": "Taxi-v3",
            "num_env_runners": args.num_env_runners,
            "train_iters": args.train_iters,
        },
    )

    config = (
        PPOConfig()
        .environment("Taxi-v3")
        .env_runners(
            num_env_runners=args.num_env_runners,
            # Taxi observations are discrete ints; one-hot flatten for the MLP.
            # Signature must be (env, spaces, device) — args may be unused/None.
            env_to_module_connector=lambda env, spaces, device: FlattenObservations(),
        )
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]))
        # Dedicated eval EnvRunner; call algo.evaluate() manually (no auto interval).
        .evaluation(evaluation_num_env_runners=1)
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        for i in range(1, args.train_iters + 1):
            result = algo.train()
            ret = _episode_return_mean(result)
            steps = result.get("num_env_steps_sampled_lifetime")
            print(
                f"iter={i}  episode_return_mean={ret:.1f}  "
                f"env_steps={steps}"
                if ret is not None
                else f"iter={i}  env_steps={steps}"
            )
            tracker.log_train(
                iteration=i, episode_return_mean=ret, env_steps=steps
            )

        eval_result = algo.evaluate()
        eval_ret = _episode_return_mean(eval_result)
        print(
            f"evaluate  episode_return_mean={eval_ret:.1f}"
            if eval_ret is not None
            else "evaluate  (no episode_return_mean)"
        )
        if eval_ret is not None:
            tracker.log_train(
                iteration=args.train_iters,
                evaluate_episode_return_mean=eval_ret,
            )
    finally:
        algo.stop()
        tracker.close()


if __name__ == "__main__":
    main()
