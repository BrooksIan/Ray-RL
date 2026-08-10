"""Train PPO on Gymnasium Taxi-v3 with Ray RLlib (new API stack)."""

from __future__ import annotations

import warnings
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


def _episode_return_mean(result: dict[str, Any]) -> float | None:
    env_runners = result.get("env_runners") or {}
    value = env_runners.get("episode_return_mean")
    return float(value) if value is not None else None


def main() -> None:
    config = (
        PPOConfig()
        .environment("Taxi-v3")
        .env_runners(
            num_env_runners=2,
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
        for i in range(1, 6):
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
