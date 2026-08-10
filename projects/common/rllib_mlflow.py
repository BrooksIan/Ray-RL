"""Optional MLflow logging for RLlib episode returns.

Enabled when ``RAY_RL_MLFLOW=1`` (or ``true`` / ``yes``). Requires ``pip install mlflow``.
"""

from __future__ import annotations

import os
from typing import Any


def _truthy(value: str | None) -> bool:
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def mlflow_enabled() -> bool:
    return _truthy(os.environ.get("RAY_RL_MLFLOW"))


class EpisodeReturnTracker:
    """No-op unless RAY_RL_MLFLOW is set and mlflow is importable."""

    def __init__(
        self,
        run_name: str,
        *,
        experiment_name: str | None = None,
        tags: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
    ) -> None:
        self._mlflow = None
        self._active = False
        if not mlflow_enabled():
            return
        try:
            import mlflow
        except ImportError as exc:
            raise SystemExit(
                "RAY_RL_MLFLOW is set but mlflow is not installed.\n"
                "  pip install mlflow\n"
                "Or unset RAY_RL_MLFLOW."
            ) from exc

        # Recent MLflow rejects the legacy filesystem store (./mlruns) unless
        # MLFLOW_ALLOW_FILE_STORE=true. Default to a local SQLite backend.
        tracking_uri = os.environ.get(
            "MLFLOW_TRACKING_URI", "sqlite:///./mlflow.db"
        )
        if tracking_uri in {"./mlruns", "file:./mlruns", "file:mlruns"} or (
            tracking_uri.startswith("file:") and "mlruns" in tracking_uri
        ):
            os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")
        experiment = experiment_name or os.environ.get(
            "RAY_RL_MLFLOW_EXPERIMENT", "ray-rllib-blueprint"
        )
        mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(experiment)
        mlflow.start_run(run_name=run_name)
        if tags:
            mlflow.set_tags(tags)
        if params:
            # MLflow params must be strings / numbers; coerce simply.
            mlflow.log_params({k: str(v) for k, v in params.items()})
        self._mlflow = mlflow
        self._active = True
        print(f"[mlflow] tracking_uri={tracking_uri} experiment={experiment} run={run_name}")

    @property
    def active(self) -> bool:
        return self._active

    def log_train(
        self,
        iteration: int,
        *,
        episode_return_mean: float | None = None,
        evaluate_episode_return_mean: float | None = None,
        env_steps: Any = None,
    ) -> None:
        if not self._active or self._mlflow is None:
            return
        metrics: dict[str, float] = {}
        if episode_return_mean is not None:
            metrics["episode_return_mean"] = float(episode_return_mean)
        if evaluate_episode_return_mean is not None:
            metrics["evaluate_episode_return_mean"] = float(
                evaluate_episode_return_mean
            )
        if env_steps is not None:
            try:
                metrics["env_steps"] = float(env_steps)
            except (TypeError, ValueError):
                pass
        if metrics:
            self._mlflow.log_metrics(metrics, step=iteration)

    def close(self) -> None:
        if not self._active or self._mlflow is None:
            return
        self._mlflow.end_run()
        self._active = False
