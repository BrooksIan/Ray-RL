"""Train offline BC from logged TicketQueue episodes (custom playground loop)."""

from __future__ import annotations

import argparse
import logging
import math
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

os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("ray.data").setLevel(logging.ERROR)
logging.getLogger("ray.data.dataset").setLevel(logging.ERROR)

from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "ticketqueue"
TRAIN_ITERS = 20

_REPO_ROOT = PROJECT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from projects.common.rllib_mlflow import EpisodeReturnTracker
from queue_env import ENV_NAME, make_ticket_queue_env


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return value_f


def _to_input_uri(path: Path) -> str:
    path = path.expanduser().resolve()
    if not path.exists():
        raise SystemExit(f"Offline input path does not exist: {path}")
    return f"local://{path.as_posix()}"


def _parse_args() -> argparse.Namespace:
    env_input = os.environ.get("RAY_RL_OFFLINE_INPUT")
    parser = argparse.ArgumentParser(
        description="Offline BC from TicketQueue episode Parquet"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(env_input) if env_input else DEFAULT_DATA_DIR,
        help=(
            "Directory of RLlib episode Parquet logs. "
            "Default: projects/custom-env-ppo/data/ticketqueue or RAY_RL_OFFLINE_INPUT"
        ),
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=TRAIN_ITERS,
        help=f"Training iterations (default: {TRAIN_ITERS})",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    data_path = args.input.expanduser()
    if not data_path.exists() or (
        data_path.is_dir() and not any(data_path.rglob("*"))
    ):
        raise SystemExit(
            f"No offline data found under {data_path}.\n"
            "Record logs first:\n"
            "  python projects/custom-env-ppo/record_queue_logs.py\n"
            "  python projects/custom-env-ppo/run_offline_pipeline.py"
        )

    register_env(ENV_NAME, make_ticket_queue_env)
    input_uri = _to_input_uri(data_path)
    print(f"[offline] training BC from {input_uri} on {ENV_NAME}")

    tracker = EpisodeReturnTracker(
        run_name="custom-env-offline-bc",
        tags={"project": "custom-env-ppo", "algorithm": "BC"},
        params={
            "env": ENV_NAME,
            "input": str(data_path.resolve()),
            "train_iters": args.train_iters,
        },
    )

    config = (
        BCConfig()
        .environment(ENV_NAME)
        .env_runners(num_env_runners=0)
        .learners(num_learners=0)
        .training(
            lr=1e-3,
            train_batch_size_per_learner=2000,
        )
        .rl_module(
            model_config=DefaultModelConfig(
                fcnet_hiddens=[64, 64],
                fcnet_activation="tanh",
                vf_share_layers=True,
            )
        )
        .offline_data(
            input_=[input_uri],
            input_read_episodes=True,
            input_read_batch_size=256,
            map_batches_kwargs={"concurrency": 1, "num_cpus": 1},
            iter_batches_kwargs={
                "prefetch_batches": 0,
                "local_shuffle_buffer_size": None,
            },
            dataset_num_iters_per_learner=5,
        )
        .evaluation(
            evaluation_interval=1,
            evaluation_num_env_runners=0,
            evaluation_duration=8,
            evaluation_duration_unit="episodes",
            evaluation_parallel_to_training=False,
            evaluation_config=BCConfig.overrides(explore=False),
        )
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        for i in range(1, args.train_iters + 1):
            result = algo.train()
            eval_block = result.get("evaluation") or {}
            eval_runners = eval_block.get("env_runners") or {}
            eval_ret = _episode_return_mean({"env_runners": eval_runners})
            if eval_ret is not None:
                print(f"iter={i}  evaluate_episode_return_mean={eval_ret:.1f}")
            else:
                print(f"iter={i}  evaluate_episode_return_mean=n/a")
            tracker.log_train(
                iteration=i, evaluate_episode_return_mean=eval_ret
            )
    finally:
        algo.stop()
        tracker.close()


if __name__ == "__main__":
    main()
