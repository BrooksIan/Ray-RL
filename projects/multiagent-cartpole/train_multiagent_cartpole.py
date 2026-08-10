"""Train multi-agent PPO on MultiAgentCartPole with Ray RLlib (new API stack)."""

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

from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.core.rl_module.default_model_config import DefaultModelConfig
from ray.rllib.examples.envs.classes.multi_agent import MultiAgentCartPole
from ray.tune.registry import register_env

NUM_AGENTS = 2
ENV_NAME = "multi-cartpole"


def _as_float(value: Any) -> float | None:
    if value is None:
        return None
    value_f = float(value)
    if math.isnan(value_f):
        return None
    return value_f


def _format_agent_returns(result: dict[str, Any]) -> str:
    env_runners = result.get("env_runners") or {}
    agent_returns = env_runners.get("agent_episode_return_mean") or {}
    module_returns = env_runners.get("module_episode_return_mean") or {}

    parts: list[str] = []
    if agent_returns:
        for agent_id, ret in sorted(agent_returns.items(), key=lambda x: str(x[0])):
            ret_f = _as_float(ret)
            if ret_f is not None:
                parts.append(f"agent[{agent_id}]={ret_f:.1f}")
    elif module_returns:
        for module_id, ret in sorted(module_returns.items(), key=lambda x: str(x[0])):
            ret_f = _as_float(ret)
            if ret_f is not None:
                parts.append(f"policy[{module_id}]={ret_f:.1f}")

    episode_ret = _as_float(env_runners.get("episode_return_mean"))
    if episode_ret is not None:
        parts.append(f"episode={episode_ret:.1f}")

    return "  ".join(parts) if parts else "returns=n/a"


def main() -> None:
    register_env(
        ENV_NAME,
        lambda _: MultiAgentCartPole({"num_agents": NUM_AGENTS}),
    )

    policies = {f"p{i}" for i in range(NUM_AGENTS)}

    config = (
        PPOConfig()
        .environment(ENV_NAME)
        .env_runners(num_env_runners=2)
        .multi_agent(
            # Separate policies = distinct controllers in one fleet.
            policies=policies,
            # Agent id 0 -> p0, agent id 1 -> p1, ...
            policy_mapping_fn=lambda agent_id, episode, **kwargs: f"p{agent_id}",
        )
        .rl_module(model_config=DefaultModelConfig(fcnet_hiddens=[64, 64]))
        .evaluation(evaluation_num_env_runners=1)
        .debugging(log_level="ERROR")
    )

    algo = config.build_algo()
    try:
        for i in range(1, 11):
            result = algo.train()
            steps = result.get("num_env_steps_sampled_lifetime")
            print(f"iter={i}  {_format_agent_returns(result)}  env_steps={steps}")

        eval_result = algo.evaluate()
        print(f"evaluate  {_format_agent_returns(eval_result)}")
    finally:
        algo.stop()


if __name__ == "__main__":
    main()
