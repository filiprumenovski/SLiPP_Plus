# Workspace bundle

Human-facing docs, generated reports, the experiment registry, and runnable examples live **under this directory**.

At the repository root, **`reports`**, **`docs`**, **`experiments`**, and **`examples`** are **symlinks** into this tree:

| Root symlink   | Points to                |
|----------------|--------------------------|
| `reports/`     | `workspace/reports/`       |
| `docs/`        | `workspace/docs/`          |
| `experiments/` | `workspace/experiments/`   |
| `examples/`    | `workspace/examples/`      |

Existing scripts, configs, and paths such as `experiments/registry.yaml` and `reports/ablation_matrix.md` resolve through those links—no path churn in code.
