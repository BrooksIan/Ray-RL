"""Train a short TicketQueue PPO policy and record episode Parquet for offline BC."""

from __future__ import annotations

import shutil
import sys
import warnings
from pathlib import Path

warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*",
    category=DeprecationWarning,
)

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.tune.registry import register_env

PROJECT_DIR = Path(__file__).resolve().parent
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from queue_env import ENV_NAME, make_ticket_queue_env

DATA_DIR = PROJECT_DIR / "data" / "ticketqueue"
CHECKPOINT_DIR = PROJECT_DIR / "data" / "ppo_checkpoint"
MODEL_CONFIG = DefaultModelConfig(
    fcnet_hiddens=[64, 64],
    fcnet_activation="tanh",
    vf_share_layers=True,
)


def _register() -> None:
    register_env(ENV_NAME, make_ticket_queue_env)


def _train_behavior_policy(train_iters: int = 10) -> str:
    """Train a decent TicketQueue policy to act as logged behavior."""
    _register()
    if CHECKPOINT_DIR.exists():
        shutil.rmtree(CHECKPOINT_DIR)

    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .env_runners(num_env_runners=1)
        .rl_module(model_config=MODEL_CONFIG)
        .debugging(log_level="ERROR")
    )
    algo = config.build_algo()
    try:
        for i in range(1, train_iters + 1):
            result = algo.train()
            env_runners = result.get("env_runners") or {}
            ret = env_runners.get("episode_return_mean")
            print(f"[record/train] iter={i}  episode_return_mean={ret}")
        return algo.save_to_path(CHECKPOINT_DIR.as_posix())
    finally:
        algo.stop()


def _record_logs(checkpoint: str, eval_iters: int = 4, episodes_per_iter: int = 25) -> Path:
    """Roll out the behavior policy and write RLlib episode Parquet files."""
    _register()
    if DATA_DIR.exists():
        shutil.rmtree(DATA_DIR)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    output_uri = f"local://{DATA_DIR.as_posix()}"

    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .env_runners(num_env_runners=1, batch_mode="complete_episodes")
        .rl_module(model_config=MODEL_CONFIG)
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_duration=episodes_per_iter,
            evaluation_duration_unit="episodes",
            evaluation_config=PPOConfig.overrides(explore=False),
        )
        .offline_data(
            output=output_uri,
            output_write_episodes=True,
            output_max_rows_per_file=25,
            output_write_remaining_data=True,
        )
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        rl_module_checkpoint = (
            Path(checkpoint)
            / COMPONENT_LEARNER_GROUP
            / COMPONENT_LEARNER
            / COMPONENT_RL_MODULE
        )
        algo.restore_from_path(
            rl_module_checkpoint.as_posix(),
            component=(
                f"{COMPONENT_LEARNER_GROUP}"
                f"/{COMPONENT_LEARNER}"
                f"/{COMPONENT_RL_MODULE}"
            ),
        )
        for i in range(1, eval_iters + 1):
            eval_result = algo.evaluate()
            eval_block = eval_result.get("evaluation") or eval_result
            env_runners = eval_block.get("env_runners") or {}
            ret = env_runners.get("episode_return_mean")
            print(f"[record/eval] iter={i}  episode_return_mean={ret}")
    finally:
        algo.stop()

    written = list(DATA_DIR.rglob("*"))
    if not written:
        raise RuntimeError(
            f"No offline files written under {DATA_DIR}. "
            "Check that msgpack-numpy is installed and recording completed."
        )
    print(f"[record] wrote {len(written)} path(s) under {DATA_DIR}")
    return DATA_DIR


def main() -> None:
    print("[record] phase 1: train short TicketQueue PPO behavior policy")
    checkpoint = _train_behavior_policy()
    print("[record] phase 2: record evaluation rollouts to Parquet episodes")
    _record_logs(checkpoint)


if __name__ == "__main__":
    main()
