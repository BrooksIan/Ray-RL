"""One-command offline smoke test: record CartPole logs, then train BC."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow `python projects/offline-marwil/run_pipeline.py` from repo root.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_cartpole_logs import main as record_main
from train_offline_marwil import main as train_main


def main() -> None:
    print("=== offline-marwil pipeline: record ===")
    record_main()

    # Tear down the record-phase Ray cluster so train-phase Ray Data is not
    # CPU-starved by leftover EnvRunners (the "Cluster resources are not enough"
    # warnings from a back-to-back run).
    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    print("=== offline-marwil pipeline: train (BC) ===")
    train_main()
    print("=== offline-marwil pipeline: done ===")


if __name__ == "__main__":
    main()
