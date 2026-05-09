# Contributing

## Development Install

Use `uv` and keep the lockfile authoritative for reproducible development environments:

```bash
uv sync --extra dev
```

For exact reproduction from the current lockfile, use:

```bash
uv sync --frozen --extra dev
```

## Tests

Run the default local gate before opening a pull request:

```bash
make test
```

For focused checks during development, run:

```bash
uv run pytest -q
uv run ruff check .
uv run ruff format --check .
make typecheck
```

`make typecheck` runs mypy against the current typed-module allowlist. Override
with `MYPY_TARGETS="src/slipp_plus/cli.py src/slipp_plus/splits.py"` as more
modules are hardened.

## Code Style

Format with Ruff:

```bash
uv run ruff format .
```

Lint with:

```bash
uv run ruff check .
```

Optional pre-commit hooks are available:

```bash
uvx pre-commit install
```

Run them manually with:

```bash
uvx pre-commit run --all-files
```

Prefer small, reviewable changes that follow the existing module boundaries. Do not rename public functions, move experiment artifacts, or alter model/science decisions without documenting the reason in `experiments/registry.yaml`.

## Commit Messages

Use concise Conventional Commit-style messages:

- `feat: add publication citation metadata`
- `fix: guard pandera import for legacy environments`
- `docs: expand CAVER feature provenance`
- `test: add day1 slow regression gate`

## Branch Model

Create a feature branch or worktree branch for publication-polish changes. Do not push directly to `main`; open a pull request and let the maintainer merge after CI passes.

Generated outputs in `processed/`, `models/`, and large structure directories should not be committed. Preserve `reports/`, `logs/`, and `experiments/registry.yaml` as the experiment audit trail.
