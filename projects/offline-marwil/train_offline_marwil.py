"""Train offline from logged CartPole episodes (BC smoke; MARWIL-ready)."""

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

# Keep the console readable: Ray Data progress lines drown out the teaching signal.
os.environ.setdefault("RAY_DATA_DISABLE_PROGRESS_BARS", "1")
logging.getLogger("ray.data").setLevel(logging.ERROR)
logging.getLogger("ray.data.dataset").setLevel(logging.ERROR)

from ray.rllib.algorithms.bc import BCConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig

PROJECT_DIR = Path(__file__).resolve().parent
DEFAULT_DATA_DIR = PROJECT_DIR / "data" / "cartpole"
TRAIN_ITERS = 20

_REPO_ROOT = PROJECT_DIR.parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from projects.common.rllib_mlflow import EpisodeReturnTracker


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
    # local:// keeps Ray Data paths portable across machines / Workbench mounts.
    return f"local://{path.as_posix()}"


def _parse_args() -> argparse.Namespace:
    env_input = os.environ.get("RAY_RL_OFFLINE_INPUT")
    parser = argparse.ArgumentParser(
        description="Offline BC from CartPole episode Parquet (skip recorder with --input)"
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path(env_input) if env_input else DEFAULT_DATA_DIR,
        help=(
            "Directory (or file) of RLlib episode Parquet logs. "
            "Default: projects/offline-marwil/data/cartpole or RAY_RL_OFFLINE_INPUT"
        ),
    )
    parser.add_argument(
        "--train-iters",
        type=int,
        default=TRAIN_ITERS,
        help=f"Training iterations (default: {TRAIN_ITERS})",
    )
    parser.add_argument(
        "--timesteps",
        action="store_true",
        help=(
            "Input is columnar timestep Parquet (not episode objects). "
            "Sets input_read_episodes=False."
        ),
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
            "Either record logs:\n"
            "  python projects/offline-marwil/record_cartpole_logs.py\n"
            "  python projects/offline-marwil/run_pipeline.py\n"
            "Or pass bring-your-own Parquet:\n"
            "  python projects/offline-marwil/train_offline_marwil.py "
            "--input /path/to/episode_parquet"
        )

    input_uri = _to_input_uri(data_path)
    read_episodes = not args.timesteps
    print(
        f"[offline] training BC from {input_uri} "
        f"(input_read_episodes={read_episodes})"
    )

    tracker = EpisodeReturnTracker(
        run_name="offline-bc",
        tags={"project": "offline-marwil", "algorithm": "BC"},
        params={
            "env": "CartPole-v1",
            "input": str(data_path.resolve()),
            "input_read_episodes": read_episodes,
            "train_iters": args.train_iters,
        },
    )

    # BC is the reliable short-demo smoke (same offline_data wiring as MARWIL).
    # For advantage-weighted MARWIL, swap BCConfig → MARWILConfig and set beta>0.
    config = (
        BCConfig()
        .environment("CartPole-v1")
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
            input_read_episodes=read_episodes,
            input_read_batch_size=256,
            # Keep a single map actor so local CPUs are not oversubscribed.
            map_batches_kwargs={"concurrency": 1, "num_cpus": 1},
            iter_batches_kwargs={
                "prefetch_batches": 0,
                "local_shuffle_buffer_size": None,
            },
            dataset_num_iters_per_learner=5,
        )
        .evaluation(
            evaluation_interval=1,
            # Local eval frees a CPU for Ray Data (avoids resource-starvation warnings).
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
