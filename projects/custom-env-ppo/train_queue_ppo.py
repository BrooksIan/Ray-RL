"""Train PPO on the custom TicketQueue playground with Ray RLlib."""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from pathlib import Path
from typing import Any

warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*",
    category=DeprecationWarning,
)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

_PROJECT_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _PROJECT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(_PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(_PROJECT_DIR))

from projects.common.rllib_mlflow import EpisodeReturnTracker
from queue_env import ENV_NAME, make_ticket_queue_env


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    return float(value) if value is not None else None


def _parse_args() -> argparse.Namespace:
    env_runners_default = int(os.environ.get("RAY_RL_NUM_ENV_RUNNERS", "2"))
    parser = argparse.ArgumentParser(
        description="PPO on custom TicketQueue-v0 playground"
    )
    parser.add_argument(
        "--num-env-runners",
        type=int,
        default=env_runners_default,
        help="RLlib EnvRunner workers (default 2 or RAY_RL_NUM_ENV_RUNNERS)",
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=12,
        help="Number of training iterations (default: 12)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    if args.num_env_runners < 0:
        raise SystemExit("--num-env-runners must be >= 0")

    register_env(ENV_NAME, make_ticket_queue_env)

    tracker = EpisodeReturnTracker(
        run_name="custom-env-ppo",
        tags={"project": "custom-env-ppo", "algorithm": "PPO"},
        params={
            "env": ENV_NAME,
            "num_env_runners": args.num_env_runners,
            "train_iters": args.train_iters,
        },
    )

    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .env_runners(num_env_runners=args.num_env_runners)
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]))
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
