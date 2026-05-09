# Tunnel Features

`v_tunnel` adds CAVER-derived tunnel geometry and lining-chemistry columns to a
base pocket table. The goal is to separate pockets that merely look elongated in
their alpha-sphere cloud from pockets that have a real exit tunnel through the
protein. That distinction is biologically relevant for the remaining fatty-acid
and steryl-ester errors: steryl esters should need longer, more buried access
paths than free fatty acids, while drug-like non-lipids often occupy compact
sites with little tunnel structure.

## Runtime Requirements

- Java must be available on `PATH`; the builder fails before processing if
  `java -version` cannot run.
- CAVER 3.0.3 is the expected binary. The default path is
  `tools/caver/caver.jar`, configurable in `configs/caver.yaml` or with
  `--caver-jar`.
- The current defaults in `configs/caver.yaml` are probe radius `0.9`,
  shell radius `3.0`, shell depth `4.0`, clustering threshold `3.5`, per-run
  timeout `60s`, structure timeout `600s`, and Java heap `768m`.

## Expected Layout

Training builds use `build_training_v_tunnel_parquet` or:

```bash
uv run python -m slipp_plus.tunnel_features training \
  --base-parquet processed/v_sterol/full_pockets.parquet \
  --source-pdbs-root data/structures/source_pdbs \
  --caver-jar tools/caver/caver.jar \
  --output processed/v_tunnel/full_pockets.parquet \
  --reports-dir reports/v_tunnel
```

For each `pdb_ligand` row group, the builder expects:

- Protein PDB: `data/structures/source_pdbs/<class>/<pdb_ligand>`.
- fpocket directory: `data/structures/source_pdbs/<class>/<stem>_out`.
- Pocket vertices: `<stem>_out/pockets/pocket<N>_vert.pqr`, where `N` is
  `matched_pocket_number`.

Holdout builds use `build_holdout_v_tunnel_parquet` or the `holdout`
subcommand. They expect `structure_id`, `matched_pocket_number`, a protein PDB
at `<structures_root>/<structure_id>.pdb`, and an fpocket output directory at
`<structures_root>/<structure_id>_out`.

## Feature Columns

The primary tunnel is the tunnel with the highest CAVER throughput for a pocket
starting point, with length as the tie-breaker. Its length, bottleneck radius,
average radius, curvature, and throughput describe the main route out of the
pocket. `tunnel_length_over_axial` compares that route with the alpha-sphere
axial length, and `tunnel_extends_beyond_pocket` marks routes that continue at
least `3 A` past the pocket extent.

The aggregate geometry columns count and summarize all tunnels assigned to the
same starting point: `tunnel_count`, `tunnel_max_length`,
`tunnel_total_length`, `tunnel_min_bottleneck`, and
`tunnel_branching_factor`. These capture whether a pocket is a single buried
channel, a branched access network, or a surface-exposed site with no meaningful
tunnel.

The lining chemistry columns use CAVER's residue table for the primary tunnel.
`tunnel_primary_hydrophobicity` is the mean Kyte-Doolittle score,
`tunnel_primary_charge` is `LYS + ARG - ASP - GLU`, and
`tunnel_primary_aromatic_fraction` is the aromatic residue fraction across PHE,
TYR, TRP, and HIS.

Three quality columns are written alongside the model features:

- `tunnel_pocket_context_present`: fpocket vertex context was present and the
  pocket centroid/axial length could be computed.
- `tunnel_caver_profile_present`: CAVER output was available and parseable for
  that pocket.
- `tunnel_has_tunnel`: at least one tunnel was assigned to that starting point.

All failure modes emit finite safe defaults so missing CAVER output is visible
to the model as zero-tunnel signal rather than as dropped rows.

## Parser Logic

CAVER output is read from the `analysis` directory. The parser accepts comma,
semicolon, or tab-delimited tables and normalizes headers before looking for
tunnel characteristics, tunnel profile points, and residue tables. It handles
both `residues.csv` and `residues.txt`.

For multi-pocket structure runs, the tunnel characteristics table must include a
starting-point column. The parser supports CAVER outputs that number starting
points either from `0` or from `1`; it selects the offset with the most hits
against the requested pocket order. If a multi-pocket run cannot be mapped back
to pockets, the code falls back to per-pocket CAVER runs.

The per-structure JSON cache records the cache version, CAVER settings,
structure metadata, rows, warnings, and optional persisted-analysis manifest
rows. Cache entries are invalidated when the cache version or CAVER settings
change.

## Quality Gates

The builder validates thresholds before work starts, runs a preflight over all
expected structure inputs, and then enforces row-level quality after the join.
Recommended thresholds are the current function defaults:

| Build | `max_missing_structure_frac` | `min_context_present_frac` | `min_profile_present_frac` |
|---|---:|---:|---:|
| Training | `0.02` | `0.98` | `0.95` |
| Holdout | `0.10` | `0.90` | `0.80` |

Use looser thresholds only for exploratory or partial batches. The build fails
if the fraction of structures with missing inputs exceeds
`max_missing_structure_frac`, if the mean `tunnel_pocket_context_present` is
below `min_context_present_frac`, or if the mean
`tunnel_caver_profile_present` is below `min_profile_present_frac`.

## Persistence Model

By default, CAVER analysis directories are temporary and only compact JSON cache
records are retained. To preserve raw CAVER analysis output for audit or later
parser debugging, pass both persistence flags:

```bash
uv run python -m slipp_plus.tunnel_features training \
  --analysis-output-root reports/v_tunnel/analysis \
  --analysis-manifest reports/v_tunnel/analysis_manifest.csv
```

`--analysis-output-root` stores copied CAVER `analysis` directories under a
stable root. `--analysis-manifest` writes a CSV manifest that maps the processed
structure/pocket rows to those persisted analysis directories. The manifest flag
requires an output root so the CSV never points at temporary directories.

The hardening work that introduced the persisted-output-first model is tracked
in commit `91bb104`.

## Failure Modes

- Missing Java or CAVER jar: the command fails before processing.
- Missing PDB, fpocket output, or pocket vertex files: preflight fails if the
  configured missing-input fraction is exceeded; otherwise affected pockets get
  safe defaults.
- CAVER timeout or non-zero exit: affected pockets get safe defaults and a
  warning in `reports/v_tunnel/build_warnings.md`.
- Missing tunnel characteristics table: affected pockets get safe defaults.
- Missing starting-point mapping in a multi-pocket run: the structure is retried
  per pocket.
- NaN or non-finite feature values: the build fails before writing the output.

Each successful build writes `build_summary.md`, `build_warnings.md`, and, when
`class_10` is present, `class_mean_sanity.md` under the selected reports
directory.
