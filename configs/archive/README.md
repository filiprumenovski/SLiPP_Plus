# configs/archive/

Closed-ablation YAML configurations. Not referenced by any active code
path; preserved so that historical experiments in
[`experiments/registry.yaml`](../../experiments/registry.yaml) remain
reproducible from the on-disk recipe that generated them.

The currently load-bearing configs live one level up:

- [`day1.yaml`](../day1.yaml) — paper-baseline reproduction.
- [`v_sterol.yaml`](../v_sterol.yaml) — v_sterol feature stack baseline.
- [`v_sterol_boundary_refactor.yaml`](../v_sterol_boundary_refactor.yaml) — boundary-head reference.
- [`v_sterol_moe.yaml`](../v_sterol_moe.yaml), [`v_sterol_pair_moe.yaml`](../v_sterol_pair_moe.yaml) — composite/MoE pipeline configs.
- [`caver.yaml`](../caver.yaml) — CAVER tunnel feature build.
