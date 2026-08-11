# Offline → online fine-tune (BC warm-start → PPO)

The usual production path: **imitate logged behavior offline**, then **carefully improve online** with a low learning rate — companion step 7.

| | |
| --- | --- |
| Entrypoint | `train_offline_to_online.py` |
| Notebook | [`offline_to_online.ipynb`](offline_to_online.ipynb) |
| Phase 1 | Behavior Cloning (`BCConfig` + policy-only RLModule) |
| Phase 2 | PPO fine-tune (`PPOConfig` + value head; loads BC weights) |
| Default env / logs | `CartPole-v1` via [`offline-marwil`](../offline-marwil/) Parquet |
| Optional env / logs | `TicketQueue-v0` via [`custom-env-ppo`](../custom-env-ppo/) Parquet |

## Why this pattern

| Phase | Role |
| --- | --- |
| Offline BC | Safe: no live exploration; copy what good operators / old controllers already did |
| Online PPO | Improve beyond the logs — but start from the BC prior so early online steps are not random |

Enterprise analogy: bootstrap from historical ticket / control logs, then allow limited live exploration to beat the clone.

## Quickstart

```bash
pip install -r projects/offline-to-online/requirements.txt

# CartPole (default): auto-records logs if missing, then BC → PPO
python projects/offline-to-online/train_offline_to_online.py

# Custom playground logs
python projects/offline-to-online/train_offline_to_online.py --env ticketqueue
```

If CartPole / TicketQueue Parquet already exists from the sibling projects, recording is skipped.

## What success looks like (CartPole)

Look for three signals in the printed summary:

1. **BC final** well above random (~20) — often ~150–250  
2. **PPO warm-start** near the BC final (clone transferred; not collapsed to ~20)  
3. **PPO after fine-tune** ≥ warm-start (online improvement; size varies by seed)

Example runs:

```text
# Run A
BC final evaluate:            205.1
PPO warm-start evaluate:      188.2
PPO after fine-tune evaluate: 437.9

# Run B
BC final evaluate:            205.5
PPO warm-start evaluate:      190.9
PPO after fine-tune evaluate: 267.9
```

Both succeed: warm-start stays near BC, and fine-tune beats the clone. Absolute fine-tune gains vary.

**Warm-start should not look random.** If PPO evaluate right after load is near ~20 on CartPole, the weight transfer failed — this project uses a shared policy trunk (`models.py`) so BC weights are a subset of the PPO module.

## How weight transfer works

```text
BCPolicyModule (encoder + π)
        │ save RLModule checkpoint
        ▼
PPOPolicyModule (same encoder + π, + zero-init value head)
        │ restore_from_path(component=…/rl_module)
        ▼
Low-lr PPO.train() online on the real env
```

See [`models.py`](models.py). Same obs/action layout for CartPole and TicketQueue (Box(4,) / Discrete(2)).

## Architecture

```text
Logged Parquet ──► BC (offline) ──► checkpoint
                                      │
                                      ▼ warm-start
                              PPO EnvRunners (online)
                                      │
                                      ▼ fine-tune
                              evaluate()  (improved policy)
```

## Requirements

- Python 3.10–3.12
- [`requirements.txt`](requirements.txt): `ray[rllib]`, Torch, Gymnasium, PyArrow, msgpack-numpy

## See also

- [Offline BC (CartPole)](../offline-marwil/) — offline-only smoke
- [Custom playground offline](../custom-env-ppo/README.md#offline-loop-logs-from-your-playground) — TicketQueue logs
- [Taxi PPO](../taxi-ppo/) — online-only cover demo
- [Project index](../README.md)
