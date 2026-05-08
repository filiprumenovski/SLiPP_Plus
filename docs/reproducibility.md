# Reproducibility

SLiPP++ reads `seed_base` from the active YAML config for every Typer CLI command that loads a config. The CLI initializes:

- Python's `random` module
- NumPy's process-level random seed
- `PYTHONHASHSEED`, if it was not already set by the caller

Model-level seeds are still passed explicitly through the training pipeline as `seed_base + iteration`, and split files are persisted under each config's processed directory. For exact dependency reproduction, install from the lockfile:

```bash
uv sync --frozen --extra dev
```

Changing the config path, feature set, split strategy, dependency lockfile, or existing persisted split files can change downstream metrics.
