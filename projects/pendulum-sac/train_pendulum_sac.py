"""Train SAC on Gymnasium Pendulum-v1 with Ray RLlib (new API stack)."""

from __future__ import annotations

import math
import warnings
from typing import Any

# Ray 2.56 still emits this from inside default RLModule construction.
warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*",
    category=DeprecationWarning,
)

from ray.rllib.algorithms.sac.sac import SACConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


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
    config = (
        SACConfig()
        .environment("Pendulum-v1")
        .env_runners(num_env_runners=1)
        .training(
            # Off-policy continuous control: reuse transitions from replay.
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 100_000,
                "alpha": 0.6,
                "beta": 0.4,
            },
            # Start learning sooner so a short smoke run shows progress.
            num_steps_sampled_before_learning_starts=1_000,
            twin_q=True,
            gamma=0.99,
            actor_lr=3e-4,
            critic_lr=3e-4,
            train_batch_size_per_learner=256,
        )
        # Pendulum episodes are 200 steps; collect enough for complete-episode metrics.
        .reporting(min_sample_timesteps_per_iteration=1_000)
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[256, 256]))
        .evaluation(evaluation_num_env_runners=1)
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        # Pendulum learns slower than CartPole; 15 iters is a short smoke run.
        for i in range(1, 16):
            result = algo.train()
            ret = _episode_return_mean(result)
            steps = result.get("num_env_steps_sampled_lifetime")
            if ret is not None:
                print(f"iter={i}  episode_return_mean={ret:.1f}  env_steps={steps}")
            else:
                print(
                    f"iter={i}  episode_return_mean=n/a  env_steps={steps}  "
                    "(no completed episodes in metric window)"
                )

        eval_result = algo.evaluate()
        eval_ret = _episode_return_mean(eval_result)
        print(
            f"evaluate  episode_return_mean={eval_ret:.1f}"
            if eval_ret is not None
            else "evaluate  episode_return_mean=n/a"
        )
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
