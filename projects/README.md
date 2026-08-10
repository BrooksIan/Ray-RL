# Projects

Each subdirectory under `projects/` is a self-contained Ray RLlib example.

## Featured

| Project | Path | Summary |
| --- | --- | --- |
| **Taxi PPO** (cover demo) | [`taxi-ppo/`](taxi-ppo/) | Modern RLlib PPO on Gymnasium `Taxi-v3` |

## Layout convention

```text
projects/<project-slug>/
  README.md           # purpose, quickstart, expected output
  requirements.txt    # pinned deps for this project
  *.py                # runnable entrypoints
  *.ipynb             # optional notebook twin
```

Optional later: `assets/`, `configs/`, `tests/` inside a project as needed.

## Add a new project

1. Copy an existing project (start from [`taxi-ppo/`](taxi-ppo/)) or create a new folder:

   ```bash
   mkdir -p projects/my-env-ppo
   ```

2. Add `README.md`, `requirements.txt`, and at least one runnable script.
3. Link it from this file and from the root [`README.md`](../README.md) projects table.
4. Keep the cover README’s **Demo / Quickstart** focused on the featured project (`taxi-ppo` today) unless you intentionally change the cover story.
