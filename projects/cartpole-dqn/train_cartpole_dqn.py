"""Train DQN on Gymnasium CartPole-v1 with Ray RLlib (new API stack)."""

from __future__ import annotations

import warnings
from typing import Any

# Ray 2.56 still emits this from inside default RLModule construction.
warnings.filterwarnings(
    "ignore",
    message=r".*RLModule\(config=\[RLModuleConfig object\]\).*",
    category=DeprecationWarning,
)

from ray.rllib.algorithms.dqn.dqn import DQNConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    return float(value) if value is not None else None


def main() -> None:
    config = (
        DQNConfig()
        .environment("CartPole-v1")
        .env_runners(num_env_runners=1)
        .training(
            # Off-policy: reuse past transitions from a prioritized replay buffer.
            replay_buffer_config={
                "type": "PrioritizedEpisodeReplayBuffer",
                "capacity": 60_000,
                "alpha": 0.5,
                "beta": 0.5,
            },
            # Start learning sooner so a short smoke run shows progress.
            num_steps_sampled_before_learning_starts=1_000,
        )
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]))
        .evaluation(evaluation_num_env_runners=1)
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        for i in range(1, 11):
            result = algo.train()
            ret = _episode_return_mean(result)
            steps = result.get("num_env_steps_sampled_lifetime")
            print(
                f"iter={i}  episode_return_mean={ret:.1f}  "
                f"env_steps={steps}"
                if ret is not None
                else f"iter={i}  env_steps={steps}"
            )

        eval_result = algo.evaluate()
        eval_ret = _episode_return_mean(eval_result)
        print(
            f"evaluate  episode_return_mean={eval_ret:.1f}"
            if eval_ret is not None
            else "evaluate  (no episode_return_mean)"
        )
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
