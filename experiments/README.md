# experiments/

Project-level experiment audit trail. Two files only:

- [`registry.yaml`](registry.yaml) — machine-readable index of every
  experiment run on this project, current-best/internal-best/superseded
  flags included. Updated by hand or by an LLM after each session.
  Source of truth for [`reports/ablation_matrix.md`](../reports/ablation_matrix.md).
- [`queued.md`](queued.md) — free-text notes on queued or explicitly
  closed work that does not have a finished registry entry yet.

Closed-ablation *runner code* (the Python files that produced these
experiments) lives in [`src/slipp_plus/experiments/`](../src/slipp_plus/experiments/) —
different folder, same word, different role.
