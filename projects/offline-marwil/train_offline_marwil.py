"""Train offline from logged CartPole episodes (BC smoke; MARWIL-ready)."""

from __future__ import annotations

import logging
import math
import os
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
DATA_DIR = PROJECT_DIR / "data" / "cartpole"
TRAIN_ITERS = 20


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return value_f


def main() -> None:
    if not DATA_DIR.exists() or not any(DATA_DIR.rglob("*")):
        raise SystemExit(
            f"No offline data found under {DATA_DIR}.\n"
            "Run first:\n"
            "  python projects/offline-marwil/record_cartpole_logs.py\n"
            "or:\n"
            "  python projects/offline-marwil/run_pipeline.py"
        )

    input_uri = f"local://{DATA_DIR.as_posix()}"
    print(f"[offline] training BC from {input_uri}")

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
            # Matches record_cartpole_logs.py (episode Parquet objects).
            input_read_episodes=True,
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
        for i in range(1, TRAIN_ITERS + 1):
            result = algo.train()
            eval_block = result.get("evaluation") or {}
            eval_runners = eval_block.get("env_runners") or {}
            eval_ret = _episode_return_mean({"env_runners": eval_runners})
            if eval_ret is not None:
                print(f"iter={i}  evaluate_episode_return_mean={eval_ret:.1f}")
            else:
                print(f"iter={i}  evaluate_episode_return_mean=n/a")
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
