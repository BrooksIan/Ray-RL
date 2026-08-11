"""Custom Gymnasium playground: single-server ticket / job queue.

Enterprise analogy: a support or ops queue where backlog costs money and
overflow misses an SLA. The agent chooses when to work the head ticket.
"""

from __future__ import annotations

from typing import Any, SupportsFloat

import gymnasium as gym
import numpy as np
from gymnasium import spaces


ENV_NAME = "TicketQueue-v0"


class TicketQueueEnv(gym.Env):
    """Discrete single-server queue with stochastic arrivals.

    Observation (float32 Box, shape=(4,)):
      0: queue length / max_queue
      1: remaining work on head ticket / max_work (0 if empty)
      2: fraction of episode elapsed
      3: 1.0 if queue is at capacity else 0.0

    Actions (Discrete 2):
      0: idle — do not process work this step
      1: work — reduce remaining work on the head ticket by 1 (if any)

    Rewards (per step):
      - ``queue_cost * queue_length`` (backlog pressure)
      - ``+complete_bonus`` when a ticket's work hits 0
      - ``-illegal_penalty`` if work is chosen on an empty queue
      - ``-overflow_penalty`` when an arrival is rejected because the queue is full
    """

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        *,
        max_queue: int = 8,
        max_work: int = 4,
        episode_len: int = 40,
        arrival_prob: float = 0.45,
        queue_cost: float = 0.25,
        complete_bonus: float = 1.0,
        illegal_penalty: float = 0.5,
        overflow_penalty: float = 2.0,
        render_mode: str | None = None,
    ) -> None:
        super().__init__()
        self.max_queue = int(max_queue)
        self.max_work = int(max_work)
        self.episode_len = int(episode_len)
        self.arrival_prob = float(arrival_prob)
        self.queue_cost = float(queue_cost)
        self.complete_bonus = float(complete_bonus)
        self.illegal_penalty = float(illegal_penalty)
        self.overflow_penalty = float(overflow_penalty)
        self.render_mode = render_mode

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(4,), dtype=np.float32
        )

        self._queue: list[int] = []
        self._t = 0
        self._np_random: np.random.Generator | None = None

    def _obs(self) -> np.ndarray:
        q_len = len(self._queue)
        head = float(self._queue[0]) if q_len else 0.0
        return np.array(
            [
                q_len / self.max_queue,
                head / self.max_work,
                self._t / self.episode_len,
                1.0 if q_len >= self.max_queue else 0.0,
            ],
            dtype=np.float32,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict[str, Any] | None = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        self._np_random = self.np_random
        self._queue = []
        self._t = 0
        # Start with one ticket so the agent has something to do immediately.
        self._queue.append(int(self._np_random.integers(1, self.max_work + 1)))
        return self._obs(), {"queue_length": len(self._queue)}

    def step(
        self, action: int
    ) -> tuple[np.ndarray, SupportsFloat, bool, bool, dict[str, Any]]:
        assert self._np_random is not None
        action = int(action)
        reward = 0.0
        completed = 0
        overflow = 0

        # Process work / idle.
        if action == 1:
            if self._queue:
                self._queue[0] -= 1
                if self._queue[0] <= 0:
                    self._queue.pop(0)
                    reward += self.complete_bonus
                    completed = 1
            else:
                reward -= self.illegal_penalty
        elif action != 0:
            raise ValueError(f"Invalid action {action}")

        # Stochastic arrival.
        if self._np_random.random() < self.arrival_prob:
            if len(self._queue) < self.max_queue:
                self._queue.append(
                    int(self._np_random.integers(1, self.max_work + 1))
                )
            else:
                reward -= self.overflow_penalty
                overflow = 1

        # Backlog cost.
        reward -= self.queue_cost * len(self._queue)

        self._t += 1
        terminated = False
        truncated = self._t >= self.episode_len
        info = {
            "queue_length": len(self._queue),
            "completed": completed,
            "overflow": overflow,
        }
        return self._obs(), float(reward), terminated, truncated, info

    def render(self) -> None:
        head = self._queue[0] if self._queue else "-"
        print(
            f"t={self._t} queue={len(self._queue)} head_work={head} "
            f"jobs={self._queue}"
        )


def make_ticket_queue_env(config: dict[str, Any] | None = None) -> TicketQueueEnv:
    """Ray ``register_env`` factory: ``register_env(name, make_ticket_queue_env)``."""
    cfg = dict(config or {})
    return TicketQueueEnv(**cfg)
