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
uv run mypy src
```

## Code Style

Format with Ruff:

```bash
uv run ruff format .
```

Lint with:

```bash
uv run ruff check .
```

Prefer small, reviewable changes that follow the existing module boundaries. Do not rename public functions, move experiment artifacts, or alter model/science decisions without documenting the reason in `RESEARCH_LOG.md` and `experiments/registry.yaml`.

## Commit Messages

Use concise Conventional Commit-style messages:

- `feat: add publication citation metadata`
- `fix: guard pandera import for legacy environments`
- `docs: expand CAVER feature provenance`
- `test: add day1 slow regression gate`

When working from `handoff.md`, reference the completed section number in the commit subject or body, for example:

```text
polish: 1.2 add citation metadata
```

## Branch Model

Create a feature branch or worktree branch for publication-polish changes. Do not push directly to `main`; open a pull request and let the maintainer merge after CI passes.

Generated outputs in `processed/`, `models/`, and large structure directories should not be committed. Preserve `reports/`, `logs/`, `RESEARCH_LOG.md`, and `experiments/registry.yaml` as the experiment audit trail.
