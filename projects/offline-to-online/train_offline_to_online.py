"""Offline BC warm-start, then short online PPO fine-tune (production path)."""

from __future__ import annotations

import argparse
import logging
import math
import os
import shutil
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
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core import (
    COMPONENT_LEARNER,
    COMPONENT_LEARNER_GROUP,
    COMPONENT_RL_MODULE,
)
from ray.rllib.core.rl_module.rl_module import RLModuleSpec
from ray.tune.registry import register_env

PROJECT_DIR = Path(__file__).resolve().parent
REPO_ROOT = PROJECT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(PROJECT_DIR) not in sys.path:
    sys.path.insert(0, str(PROJECT_DIR))

from models import BCPolicyModule, PPOPolicyModule
from projects.common.rllib_mlflow import EpisodeReturnTracker

CARTPOLE_DATA = REPO_ROOT / "projects" / "offline-marwil" / "data" / "cartpole"
TICKET_DATA = REPO_ROOT / "projects" / "custom-env-ppo" / "data" / "ticketqueue"
BC_CHECKPOINT_DIR = PROJECT_DIR / "data" / "bc_checkpoint"


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    if value is None:
        return None
    value_f = float(value)
    return None if math.isnan(value_f) else value_f


def _eval_return(result: dict[str, Any]) -> float | None:
    block = result.get("evaluation") or result
    return _episode_return_mean({"env_runners": block.get("env_runners") or {}})


def _setup_env(env_key: str) -> tuple[str, Path]:
    if env_key == "cartpole":
        return "CartPole-v1", CARTPOLE_DATA
    if env_key == "ticketqueue":
        custom_dir = REPO_ROOT / "projects" / "custom-env-ppo"
        if str(custom_dir) not in sys.path:
            sys.path.insert(0, str(custom_dir))
        from queue_env import ENV_NAME, make_ticket_queue_env

        register_env(ENV_NAME, make_ticket_queue_env)
        return ENV_NAME, TICKET_DATA
    raise SystemExit(f"Unknown --env {env_key!r} (use cartpole or ticketqueue)")


def _ensure_data(env_key: str, data_dir: Path) -> None:
    if data_dir.exists() and any(data_dir.rglob("*")):
        return
    print(f"[offline-to-online] no logs under {data_dir}; recording now…")
    if env_key == "cartpole":
        sys.path.insert(0, str(REPO_ROOT / "projects" / "offline-marwil"))
        from record_cartpole_logs import main as record_main

        record_main()
    else:
        sys.path.insert(0, str(REPO_ROOT / "projects" / "custom-env-ppo"))
        from record_queue_logs import main as record_main

        record_main()
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass
    if not data_dir.exists() or not any(data_dir.rglob("*")):
        raise SystemExit(f"Recording finished but still no data under {data_dir}")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="BC warm-start from logs, then PPO fine-tune online"
    )
    parser.add_argument(
        "--env",
        choices=("cartpole", "ticketqueue"),
        default="cartpole",
        help="Playground / data source (default: cartpole)",
    )
    parser.add_argument("--bc-iters", type=int, default=15)
    parser.add_argument("--ppo-iters", type=int, default=10)
    parser.add_argument(
        "--skip-record",
        action="store_true",
        help="Fail if logs are missing instead of auto-recording",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    env_name, data_dir = _setup_env(args.env)
    if args.skip_record:
        if not data_dir.exists() or not any(data_dir.rglob("*")):
            raise SystemExit(
                f"No offline data under {data_dir}.\n"
                "Record first or omit --skip-record to auto-record."
            )
    else:
        _ensure_data(args.env, data_dir)

    input_uri = f"local://{data_dir.resolve().as_posix()}"
    print(f"[offline-to-online] env={env_name} logs={input_uri}")

    tracker = EpisodeReturnTracker(
        run_name=f"offline-to-online-{args.env}",
        tags={"project": "offline-to-online", "algorithm": "BC→PPO"},
        params={
            "env": env_name,
            "bc_iters": args.bc_iters,
            "ppo_iters": args.ppo_iters,
        },
    )

    # --- Phase 1: offline BC ---
    print("=== phase 1: offline BC warm-start ===")
    bc_config = (
        BCConfig()
        .environment(env_name)
        .env_runners(num_env_runners=0)
        .learners(num_learners=0)
        .training(lr=1e-3, train_batch_size_per_learner=2000)
        .rl_module(rl_module_spec=RLModuleSpec(module_class=BCPolicyModule))
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
    bc = bc_config.build_algo()
    bc_eval = None
    try:
        for i in range(1, args.bc_iters + 1):
            result = bc.train()
            ret = _eval_return(result)
            if ret is not None:
                print(f"[bc] iter={i}  evaluate_episode_return_mean={ret:.1f}")
            else:
                print(f"[bc] iter={i}  evaluate_episode_return_mean=n/a")
            tracker.log_train(iteration=i, evaluate_episode_return_mean=ret)
        bc_eval = _eval_return(bc.evaluate())
        if bc_eval is not None:
            print(f"[bc] final_evaluate={bc_eval:.1f}")
        else:
            print("[bc] final_evaluate=n/a")
        if BC_CHECKPOINT_DIR.exists():
            shutil.rmtree(BC_CHECKPOINT_DIR)
        BC_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
        ckpt_path = bc.save_to_path(BC_CHECKPOINT_DIR.resolve().as_posix())
    finally:
        bc.stop()

    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    rl_module_checkpoint = (
        Path(ckpt_path)
        / COMPONENT_LEARNER_GROUP
        / COMPONENT_LEARNER
        / COMPONENT_RL_MODULE
    )

    # --- Phase 2: online PPO fine-tune ---
    print("=== phase 2: online PPO fine-tune (low lr) ===")
    # Re-register TicketQueue after Ray restart.
    env_name, _ = _setup_env(args.env)

    ppo_config = (
        PPOConfig()
        .environment(env_name)
        .env_runners(num_env_runners=1)
        .training(
            # Low LR to avoid wiping the BC prior too quickly.
            lr=3e-5,
            num_epochs=4,
            vf_loss_coeff=0.01,
        )
        .rl_module(
            rl_module_spec=RLModuleSpec(
                module_class=PPOPolicyModule,
                load_state_path=rl_module_checkpoint.as_posix(),
            )
        )
        .evaluation(
            evaluation_num_env_runners=1,
            evaluation_duration=8,
            evaluation_duration_unit="episodes",
        )
        .debugging(log_level="ERROR")
    )
    ppo = ppo_config.build_algo()
    try:
        ppo.restore_from_path(
            rl_module_checkpoint.as_posix(),
            component=(
                f"{COMPONENT_LEARNER_GROUP}"
                f"/{COMPONENT_LEARNER}"
                f"/{COMPONENT_RL_MODULE}"
            ),
        )
        if hasattr(ppo, "env_runner_group") and ppo.env_runner_group is not None:
            ppo.env_runner_group.sync_weights(
                from_worker_or_learner_group=ppo.learner_group
            )

        warm = _eval_return(ppo.evaluate())
        if warm is not None:
            print(f"[ppo] warm_start_evaluate={warm:.1f}")
            tracker.log_train(
                iteration=args.bc_iters,
                evaluate_episode_return_mean=warm,
            )
        else:
            print("[ppo] warm_start_evaluate=n/a")

        for i in range(1, args.ppo_iters + 1):
            result = ppo.train()
            ret = _episode_return_mean(result)
            if ret is not None:
                print(f"[ppo] iter={i}  episode_return_mean={ret:.1f}")
            else:
                print(f"[ppo] iter={i}  episode_return_mean=n/a")
            tracker.log_train(iteration=args.bc_iters + i, episode_return_mean=ret)

        final = _eval_return(ppo.evaluate())
        if final is not None:
            print(f"[ppo] final_evaluate={final:.1f}")
            tracker.log_train(
                iteration=args.bc_iters + args.ppo_iters,
                evaluate_episode_return_mean=final,
            )
        else:
            print("[ppo] final_evaluate=n/a")

        print("--- summary ---")
        print(
            f"BC final evaluate:            {bc_eval:.1f}"
            if bc_eval is not None
            else "BC final evaluate:            n/a"
        )
        print(
            f"PPO warm-start evaluate:      {warm:.1f}"
            if warm is not None
            else "PPO warm-start evaluate:      n/a"
        )
        print(
            f"PPO after fine-tune evaluate: {final:.1f}"
            if final is not None
            else "PPO after fine-tune evaluate: n/a"
        )
    finally:
        ppo.stop()
        tracker.close()


if __name__ == "__main__":
    main()
