"""Record TicketQueue logs, then train offline BC (custom playground loop)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from record_queue_logs import main as record_main
from train_offline_queue_bc import main as train_main


def main() -> None:
    print("=== custom-env-ppo offline pipeline: record ===")
    record_main()

    try:
        import ray

        if ray.is_initialized():
            ray.shutdown()
    except Exception:
        pass

    print("=== custom-env-ppo offline pipeline: train (BC) ===")
    train_main()
    print("=== custom-env-ppo offline pipeline: done ===")


if __name__ == "__main__":
    main()
