# Tier 3 Tunnel Features — Agent Brief

**Status:** ready to execute. Independent of the ongoing sterol-ID sprint.
**Owner:** whoever the next coding agent is.
**Compute:** laptop-class. 8-core parallel, ~30-60 minutes wall for the full training set. No HPC.
**Hypothesis:** a proper Voronoi-graph tunnel computation (CAVER) applied per pocket will resolve the remaining PLM-vs-STE and OLA-vs-{MYR,PLM} confusion that our alpha-sphere-only geometric features can't see. Steryl esters need ~2× the tunnel length of fatty acids; drug-like ligands need none. Our current features conflate pocket-extent with tunnel-length.

**Non-goal:** replace the `v_sterol` ensemble + PLM/STE tiebreaker stack. Tunnel features are *additive*; the winning config becomes `v_tunnel` (= `v_sterol` + 15 tunnel columns) + the same ensemble + the same tiebreaker.

---

## 0. Success criteria (locked before starting)

Commit the success/kill thresholds up-front so we don't p-hack later.

**Primary win condition (any one triggers "ship it"):**
- STE F1 on the test split rises by ≥ +0.015 over the current winning config (`v_sterol` + PLM/STE tiebreaker at margin=0.99, STE F1 = 0.444 ± 0.107).
- 10-class macro-F1 rises by ≥ +0.005 (current: 0.738 ± 0.015).
- Any tunnel-derived feature lands in the top-10 XGB gain for the PLM/STE tiebreaker iter-0 dump.

**Kill conditions (any one triggers "abandon, keep code, revert config"):**
- STE F1 regresses by > 0.01.
- Binary F1 on the paper's lipid-vs-rest collapse regresses by > 0.005.
- CAVER fails (empty or error) on > 15% of training structures.
- Total end-to-end compute (feature build + retrain + tiebreaker) exceeds 6 hours on an 8-core laptop.

All numbers reported as mean ± std across the existing 25 stratified shuffle splits in `processed/splits/seed_*.parquet`. Do not rebuild splits.

---

## 1. Context — what's already in the repo

Do not re-read everything; this section is the minimum context needed.

- Winning config today (`reports/sterol_id_sprint.md`): `v_sterol` features + RF/XGB/LGBM probability-average ensemble + PLM-vs-STE binary XGB tiebreaker with margin=0.99. STE F1 = 0.444, macro-F1 = 0.738.
- Feature registry lives in `src/slipp_plus/constants.py:FEATURE_SETS`. `v_sterol` = paper's 17 + 20 AA counts + 12 aromatic/aliphatic shell features + 32 chemistry-refined shells + 6 alpha-sphere PCA/burial columns. 87 numeric columns total.
- Training parquet: `processed/v_sterol/full_pockets.parquet`, 15,219 rows. Keyed by `pdb_ligand` (e.g. `CLR/pdb3GKS.pdb`) + `matched_pocket_number`.
- Structures + fpocket outputs: `data/structures/source_pdbs/<CLASS>/<stem>.pdb` and `data/structures/source_pdbs/<CLASS>/<stem>_out/pockets/pocket{N}_{atm.pdb,vert.pqr}`.
- Helper patterns to reuse: `src/slipp_plus/sterol_features.py` and `src/slipp_plus/plm_ste_features.py` are the canonical templates. Copy their ProcessPoolExecutor worker shape, their empty-row safe defaults, their warning-collection pattern, their CLI surface.
- Training entrypoint: `uv run python -m slipp_plus train --config configs/<feature_set>.yaml`. Reads `feature_set`, reads `feature_columns()` off settings, writes `processed/<set>/predictions/test_predictions.parquet` + `models/<set>/{rf,xgb,lgbm}_multiclass.joblib`.
- Tiebreaker: `src/slipp_plus/plm_ste_tiebreaker.py` already accepts `--prefix` and `--model-bundle`. Use it as-is.

**DO NOT** modify `sterol_features.py`, `plm_ste_features.py`, `plm_ste_tiebreaker.py`, `sterol_tiebreaker.py`, `ensemble.py`, `train.py`, `schemas.py`, `splits.py`, or `reference/`. Everything new lives in its own module and its own parquet tree.

---

## 2. Environment — CAVER install

### 2.1 Primary tool: CAVER 3.0.3 (Java JAR)

```bash
# We already have Java 17 from the P2Rank sprint — verify:
java -version       # expect "openjdk 17.x" or newer

# Download CAVER 3.0.3:
mkdir -p tools/caver
cd tools/caver
curl -L -o caver-3.0.3.zip \
  "https://www.caver.cz/download/caver_3_0_3.zip"
unzip caver-3.0.3.zip
# Expect a folder containing caver.jar + lib/ + libexec/ + settings/default.txt
# Verify invocation:
java -jar caver.jar -help | head -20
```

If `caver.cz` is down or the URL rotates, mirror from `https://github.com/loschmidt/caver/releases` (look for 3.0.3 / 3.0.2). Pin whichever version you use; do not silently upgrade.

**Persist the absolute path** to `caver.jar` in `configs/caver.yaml` (new file). Every downstream module reads that.

### 2.2 Optional second backend: MOLE 2.5

Only needed if Section 7 (cross-validation) flags CAVER output as suspicious. Skip during the initial pass. If needed:

```bash
# MOLE is .NET; easiest path on macOS:
brew install dotnet
# Then download MOLE 2.5 from https://webchemdev.ncbr.muni.cz/Platform/AppsBin/
# Invoke with `dotnet MOLE.dll`. Output is JSON.
```

### 2.3 Sanity smoke test

```bash
# Pick one CLR structure with a known long cholesterol tunnel (START-domain):
uv run python -c "
import subprocess, tempfile, pathlib
caver = pathlib.Path('tools/caver/caver.jar').resolve()
pdb   = pathlib.Path('data/structures/source_pdbs/CLR/pdb3GKS.pdb').resolve()
assert caver.exists() and pdb.exists()
print('caver ok:', caver)
print('pdb ok:', pdb)
"
```

If both paths resolve, move on.

---

## 3. Module design

Create `src/slipp_plus/tunnel_features.py`. Mirror the structure of `src/slipp_plus/plm_ste_features.py`: public extractor + ProcessPoolExecutor worker + `build_training_v_tunnel_parquet` + `build_holdout_v_tunnel_parquet` + argparse `main()`.

### 3.1 Public API

```python
TUNNEL_FEATURES_15: list[str] = [
    "tunnel_count",
    "tunnel_primary_length",
    "tunnel_primary_bottleneck_radius",
    "tunnel_primary_avg_radius",
    "tunnel_primary_curvature",
    "tunnel_primary_throughput",
    "tunnel_primary_hydrophobicity",
    "tunnel_primary_charge",
    "tunnel_primary_aromatic_fraction",
    "tunnel_max_length",
    "tunnel_total_length",
    "tunnel_min_bottleneck",
    "tunnel_branching_factor",
    "tunnel_length_over_axial",
    "tunnel_extends_beyond_pocket",
]

def extract_pocket_tunnel_features(
    protein_pdb_path: Path,
    pocket_centroid: np.ndarray,
    pocket_axial_length: float,
    caver_jar: Path,
    *,
    probe_radius: float = 0.9,
    shell_radius: float = 3.0,
    shell_depth: float = 4.0,
    clustering_threshold: float = 3.5,
    timeout_s: int = 45,
) -> dict[str, float]: ...

def build_training_v_tunnel_parquet(
    base_parquet: Path,            # default processed/v_sterol/full_pockets.parquet
    source_pdbs_root: Path,        # default data/structures/source_pdbs
    caver_jar: Path,               # from configs/caver.yaml
    output_path: Path,             # default processed/v_tunnel/full_pockets.parquet
    workers: int = 8,
) -> dict[str, object]: ...

def build_holdout_v_tunnel_parquet(
    base_parquet: Path,
    structures_root: Path,
    caver_jar: Path,
    output_path: Path,
    workers: int = 8,
) -> dict[str, object]: ...
```

### 3.2 Worker pattern — one CAVER run per structure, N pocket starting-points

**Critical design choice.** Do not run CAVER once per pocket; run it once per structure with all of that structure's pocket centroids supplied as multiple starting points. CAVER handles multi-point starting-point sets natively and this is 5-10× cheaper.

Per-structure worker outline (do not write the whole thing here — this is just the shape):

```
def _process_structure(task):
    protein_pdb = task["protein_pdb"]
    pocket_centroids = task["pocket_centroids"]  # dict[pocket_number] -> (x,y,z)
    pocket_axial = task["pocket_axial_lengths"]  # dict[pocket_number] -> float
    caver_jar = task["caver_jar"]

    with TemporaryDirectory() as tmp:
        # Write caver config.txt pointing at:
        #   - protein_pdb as input
        #   - starting_point_coordinates for each pocket (CAVER allows multiple)
        #   - output_dir = tmp/caver_out
        #   - probe_radius / shell_radius / etc. from kwargs
        # Invoke:
        #   subprocess.run(["java", "-jar", str(caver_jar), config_path],
        #                   timeout=timeout_s * n_pockets, capture_output=True)
        # Parse:
        #   tmp/caver_out/analysis/tunnels.csv           (per-tunnel summary)
        #   tmp/caver_out/analysis/tunnel_profiles.csv   (per-tunnel radius profile)
        #   tmp/caver_out/analysis/residues.csv          (lining residues per tunnel)
        # For each pocket centroid, associate all tunnels whose
        #   starting_point_index matches that centroid's submission index.
        # Compute the 15 features from the associated tunnel(s).

    return {"rows": [...], "warnings": [...]}
```

Key implementation details:
- **One temp dir per structure run.** CAVER writes ~20 MB of intermediate files per run. Use `tempfile.TemporaryDirectory` so each worker is isolated. Do not let workers share output dirs.
- **Subprocess isolation.** Use `subprocess.run(..., check=False, timeout=...)`. Catch `TimeoutExpired` and emit a warning with safe-default features for every pocket in that structure.
- **CAVER config file.** Generate in the tmp dir. Reference docs: https://loschmidt.chemi.muni.cz/caver/wiki/index.php/Configuration (mirror at `reference/caver_config.md` — write this during setup).
- **Starting point format.** Each pocket centroid becomes one line:
  `starting_point_coordinates  <x> <y> <z>`. Track submission order so tunnel outputs (indexed 0..N-1) map back to pocket numbers.

### 3.3 Feature extraction from CAVER output

CAVER emits (at `out_dir/analysis/`):
- `tunnels.csv` — one row per tunnel with: `Tunnel cluster`, `Starting point`, `Length`, `Bottleneck radius`, `Curvature`, `Throughput`, `Avg R`.
- `tunnel_profiles.csv` — per-tunnel point-by-point `(distance_from_origin, R)` profile. Use to double-check bottleneck if needed.
- `residues.csv` — lining residues per tunnel cluster with residue name + number + chain.

**Feature derivations (exact):**

| feature | derivation |
|---|---|
| `tunnel_count` | # distinct tunnels associated with this pocket's starting point |
| `tunnel_primary_length` | `Length` of tunnel with highest `Throughput` (primary tunnel); 0.0 if none |
| `tunnel_primary_bottleneck_radius` | `Bottleneck radius` of primary; 0.0 if none |
| `tunnel_primary_avg_radius` | `Avg R` of primary; 0.0 if none |
| `tunnel_primary_curvature` | `Curvature` of primary; 1.0 if none (straight / undefined) |
| `tunnel_primary_throughput` | `Throughput` of primary; 0.0 if none |
| `tunnel_primary_hydrophobicity` | mean Kyte-Doolittle score of primary's lining residues; 0.0 if none. Use standard KD table (hard-code a dict in `tunnel_features.py`). |
| `tunnel_primary_charge` | (# LYS + # ARG − # ASP − # GLU) in primary lining; 0.0 if none |
| `tunnel_primary_aromatic_fraction` | (# PHE + # TYR + # TRP + # HIS) / total lining count; 0.0 if empty |
| `tunnel_max_length` | max `Length` across all this pocket's tunnels; 0.0 if none |
| `tunnel_total_length` | sum `Length` across all this pocket's tunnels; 0.0 if none |
| `tunnel_min_bottleneck` | min `Bottleneck radius` across all tunnels; 0.0 if none |
| `tunnel_branching_factor` | # tunnels starting from this point that share their first 3 Å of path with another tunnel from the same point. Compute by comparing `tunnel_profiles.csv` entries for tunnels with the same `Starting point` — if two tunnels have matching coordinate sequences for the first 3 Å, they share a junction. 0 if only one tunnel. |
| `tunnel_length_over_axial` | `tunnel_primary_length / (pocket_axial_length + 1e-6)`. Values > 1 indicate the tunnel extends the pocket outward. Clip to `[0, 20]`. |
| `tunnel_extends_beyond_pocket` | 1 if `tunnel_primary_length > pocket_axial_length + 3.0` else 0 (int). |

All features must be **numeric, finite, no NaN**. Replace any `inf` from division-by-zero with 0.0 (counts) or 1.0 (ratios that default to "identity").

### 3.4 Failure handling (mandatory — the pipeline will hit these)

Expected failure modes and the required behavior:

| failure | detection | action |
|---|---|---|
| CAVER timeout | `subprocess.TimeoutExpired` | log warning, emit safe-default 15-tuple for every pocket in the structure, continue |
| CAVER non-zero exit | `returncode != 0` | same as timeout |
| CAVER produced no `tunnels.csv` | file missing | same as timeout |
| Pocket has zero tunnels starting from its centroid | no rows in `tunnels.csv` for that starting-point index | emit the zero-tunnel safe default (all zeros, `curvature=1.0`, `aromatic_fraction=0.0`). This is *not* a failure — it's a signal (surface-exposed pocket). |
| Pocket centroid outside protein envelope | CAVER errors with "starting point not found inside protein" | retry once with `probe_radius=0.7`. If still failing, safe-default + warning. |
| Multi-chain PDB with HETATM waters | CAVER crashes on some non-standard residues | preprocess: strip HETATM lines except for ligand? No — fpocket already includes HETATM ligand context. Strip only `HOH` waters in a one-time preprocessing pass. |

**Safe defaults** (for any failure case):
```python
{
    "tunnel_count": 0,
    "tunnel_primary_length": 0.0,
    "tunnel_primary_bottleneck_radius": 0.0,
    "tunnel_primary_avg_radius": 0.0,
    "tunnel_primary_curvature": 1.0,
    "tunnel_primary_throughput": 0.0,
    "tunnel_primary_hydrophobicity": 0.0,
    "tunnel_primary_charge": 0.0,
    "tunnel_primary_aromatic_fraction": 0.0,
    "tunnel_max_length": 0.0,
    "tunnel_total_length": 0.0,
    "tunnel_min_bottleneck": 0.0,
    "tunnel_branching_factor": 0,
    "tunnel_length_over_axial": 0.0,
    "tunnel_extends_beyond_pocket": 0,
}
```

Collect all warnings; dump a summary at `reports/v_tunnel/build_warnings.md` with counts by class.

---

## 4. Parallelization

```python
# In build_training_v_tunnel_parquet:
#   groups = base.groupby("pdb_ligand") -> one task per structure
#   with ProcessPoolExecutor(max_workers=workers) as ex:
#       results = list(ex.map(_process_structure, tasks))
```

- Default `workers=8`. On beefier boxes, accept `--workers 16` from CLI.
- Structures average ~8 pockets each. 1,780 structures / 8 workers ≈ 222 tasks/worker. Per structure CAVER runtime ≈ 5-15 s. Expected wall: 20-45 minutes.
- Each worker spawns a Java subprocess. Set a small JVM heap: `java -Xmx512m -jar caver.jar config.txt`. 8 × 512 MB = 4 GB peak, fits anywhere.
- **Do not** use `ThreadPoolExecutor`. Each Java subprocess holds its own memory; threads gain nothing.

---

## 4.1 Reliability guardrails (implemented)

The builder now fails fast instead of silently writing fallback-heavy outputs when
the structure root is wrong or incomplete.

- **Preflight input validation:** before CAVER starts, every structure is checked
  for `<stem>.pdb`, `<stem>_out/`, and `<stem>_out/pockets/`.
- **Hard failure threshold:** if the fraction of structures with missing inputs is
  above `--max-missing-structure-frac`, the run aborts immediately.
- **Post-build quality gates:** output is rejected if either
  `tunnel_pocket_context_present` or `tunnel_caver_profile_present` means fall
  below thresholds (`--min-context-present-frac`,
  `--min-profile-present-frac`).

### Common holdout path pitfall

For holdout builds, `--structures-root` must point to the folder that directly
contains `<structure_id>.pdb` and `<structure_id>_out/pockets/`, for example:

```bash
--structures-root processed/v49/structures/alphafold_holdout
```

Pointing to `processed/v_sterol/...` usually fails because those structure trees
are not created by ingest/training alone.

### Persist CAVER CSV outputs for later extraction

If you need reusable CAVER output tables (instead of temp-only parsing), run with:

- `--analysis-output-root <dir>` to persist each structure's `analysis/` files.
- `--analysis-manifest <path.csv>` to emit pocket mapping rows:
  key column (`pdb_ligand` or `structure_id`), `matched_pocket_number`,
  `starting_point_index`, `pocket_axial_length`, `analysis_dir`.

---

## 5. Integration into the feature pipeline

### 5.1 Extend `constants.py`

```python
# In src/slipp_plus/constants.py, add AFTER the STEROL_* definitions:
TUNNEL_FEATURES_15: list[str] = [ ... see 3.1 ... ]

# Add to FEATURE_SETS:
FEATURE_SETS["v_tunnel"] = (
    SELECTED_17
    + AA20
    + AROMATIC_ALIPHATIC_12
    + STEROL_CHEMISTRY_SHELL_COLS
    + POCKET_GEOMETRY_COLS
    + TUNNEL_FEATURES_15
)
```

### 5.2 Extend `config.py`

```python
FeatureSet = Literal[
    "v14", "v14+v22", "v14+aa", "v14+v22+aa",
    "v49", "v61", "v_sterol", "v_plm_ste", "v_tunnel",
]
```

### 5.3 Create `configs/v_tunnel.yaml`

Clone `configs/v_sterol.yaml`, change:
```yaml
feature_set: v_tunnel
paths:
  processed_dir: processed/v_tunnel
  models_dir: models/v_tunnel
  reports_dir: reports/v_tunnel
```

### 5.4 Create `configs/caver.yaml`

```yaml
caver_jar: tools/caver/caver.jar
probe_radius: 0.9
shell_radius: 3.0
shell_depth: 4.0
clustering_threshold: 3.5
timeout_s: 45
```

Read from this inside `tunnel_features.py` so parameters live in one place.

---

## 6. Execution plan — exact commands

Run these in order. Each step must finish cleanly before proceeding.

```bash
# (1) Install CAVER — see Section 2.
java -jar tools/caver/caver.jar -help | head -1

# (2) Smoke test on a single structure (pick a long-tunnel CLR case):
uv run python -c "
from pathlib import Path
from slipp_plus.tunnel_features import extract_pocket_tunnel_features
import numpy as np
out = extract_pocket_tunnel_features(
    protein_pdb_path=Path('data/structures/source_pdbs/CLR/pdb3GKS.pdb'),
    pocket_centroid=np.array([0.0, 0.0, 0.0]),  # replace with actual centroid
    pocket_axial_length=15.0,
    caver_jar=Path('tools/caver/caver.jar'),
)
print(out)
"
# Expect: tunnel_primary_length > 0, tunnel_count >= 1 for a known pocket.
# If it fails, debug BEFORE launching the full build.

# (3) Build the v_tunnel parquet (full training set, parallel):
uv run python -m slipp_plus.tunnel_features training \
    --base-parquet processed/v_sterol/full_pockets.parquet \
    --source-pdbs-root data/structures/source_pdbs \
    --caver-jar tools/caver/caver.jar \
    --output processed/v_tunnel/full_pockets.parquet \
    --workers 8

# Expected: 15,219 rows, 15 new columns, 0 NaN, warnings list printed.
# If warnings > 15% of structures, STOP and investigate.

# (4) Retrain 3 multiclass heads on v_tunnel:
uv run python -m slipp_plus train --config configs/v_tunnel.yaml
# Expected: processed/v_tunnel/predictions/test_predictions.parquet
#           models/v_tunnel/{rf,xgb,lgbm}_multiclass.joblib

# (5) Run the winning-config overlay: PLM/STE tiebreaker on top.
uv run python -m slipp_plus.plm_ste_tiebreaker \
    --full-pockets processed/v_tunnel/full_pockets.parquet \
    --predictions  processed/v_tunnel/predictions/test_predictions.parquet \
    --splits-dir   processed/splits \
    --output       reports/v_tunnel/plm_ste_tiebreaker_metrics.md \
    --tiebreaker-predictions processed/v_tunnel/predictions/plm_ste_tiebreaker_predictions.parquet \
    --ensemble-predictions   processed/v_tunnel/predictions/ensemble_predictions.parquet \
    --overall-report         reports/v_tunnel/overall_metrics.md \
    --model-bundle models/v_tunnel/xgb_multiclass.joblib \
    --prefix v_tunnel \
    --margin 0.99 \
    --workers 8

# (6) Holdouts — only if (5) crosses the win threshold.
uv run python -m slipp_plus.tunnel_features holdout \
    --base-parquet processed/v_sterol/apo_pdb_holdout.parquet \
    --structures-root processed/v_sterol/structures/apo_pdb_holdout \
    --caver-jar tools/caver/caver.jar \
    --output processed/v_tunnel/apo_pdb_holdout.parquet \
    --workers 8

uv run python -m slipp_plus.tunnel_features holdout \
    --base-parquet processed/v_sterol/alphafold_holdout.parquet \
    --structures-root processed/v_sterol/structures/alphafold_holdout \
    --caver-jar tools/caver/caver.jar \
    --output processed/v_tunnel/alphafold_holdout.parquet \
    --workers 8

uv run python -m slipp_plus v49-holdouts --config configs/v_tunnel.yaml   # or equivalent holdout eval CLI
```

---

## 7. Validation gates

The subagent must pass every gate in order before declaring the build "complete."

### 7.1 Parquet integrity (hard gate)

```python
import pandas as pd
df = pd.read_parquet("processed/v_tunnel/full_pockets.parquet")
assert len(df) == 15219, f"row drift: {len(df)}"
for c in TUNNEL_FEATURES_15:
    assert c in df.columns, f"missing column: {c}"
    assert df[c].notna().all(), f"{c} has NaNs"
    assert np.isfinite(df[c]).all(), f"{c} has inf"
```

### 7.2 Class-mean sanity (hard gate)

```python
means = df.groupby("class_10")[TUNNEL_FEATURES_15].mean()
# Biochemistry predictions that MUST hold:
assert means.loc["CLR", "tunnel_primary_length"] > means.loc["COA", "tunnel_primary_length"], \
    "CLR tunnels should be longer than COA (drug-like) tunnels"
assert means.loc["STE", "tunnel_primary_length"] > means.loc["PP",  "tunnel_primary_length"], \
    "STE tunnels should be longer than pseudo-pocket tunnels"
# Soft check — expected direction but not hard gate:
# means.loc["STE", "tunnel_primary_length"] should be >= means.loc["PLM", "tunnel_primary_length"]
# If this fails, document it; do not kill the run.
```

Dump the class-mean table to `reports/v_tunnel/class_mean_sanity.md`.

### 7.3 Cross-validation with MOLE (optional, recommended)

Only run if you have > 2 hours of budget left. Sample 100 structures stratified by class (10 per class), run MOLE on the same starting points, compute Pearson `r` between CAVER and MOLE `tunnel_primary_length`. Expected `r > 0.75`. If `r < 0.50`, CAVER output is suspect — investigate starting-point convention or probe-radius calibration.

### 7.4 Feature importance gate (soft)

After step (5), inspect `reports/v_tunnel/plm_ste_tiebreaker_metrics.md` → top-15 gain table. At least one of `tunnel_primary_length`, `tunnel_length_over_axial`, `tunnel_extends_beyond_pocket`, `tunnel_min_bottleneck` should appear in the top 10. If none appear, tunnel features are redundant with existing geometry — tunnel features have failed the "are they additive" test.

### 7.5 Metric gate (primary success condition)

```
v_sterol + PLM/STE tb:   macro-F1 0.738, STE F1 0.444      <-- baseline
v_tunnel + PLM/STE tb:   macro-F1 ≥ 0.743, STE F1 ≥ 0.459  <-- target
```

If both hit, update `reports/sterol_id_sprint.md` with a Round 3 section analogous to Round 2. If only STE improves, still ship it — STE is the open item. If only macro-F1 improves but STE regresses > 0.01, kill and investigate.

---

## 8. Deliverables

The subagent must produce all of the following before reporting completion:

**Code:**
- `src/slipp_plus/tunnel_features.py` (new)
- `src/slipp_plus/constants.py` (modified — add `TUNNEL_FEATURES_15` and `v_tunnel` entry)
- `src/slipp_plus/config.py` (modified — extend `FeatureSet` Literal)
- `configs/v_tunnel.yaml` (new)
- `configs/caver.yaml` (new)
- `tools/caver/caver.jar` (+ lib/, settings/, as installed)
- `tests/test_tunnel_features.py` with at least:
  - parse-tunnels-csv smoke test (using a canned CAVER output fixture)
  - feature-dict-completeness test (every key in `TUNNEL_FEATURES_15` is returned, all finite)
  - safe-default-on-failure test (simulate CAVER exit code 1, assert safe defaults)

**Data:**
- `processed/v_tunnel/full_pockets.parquet` — 15,219 rows × (87 + 15) columns.
- `processed/v_tunnel/predictions/test_predictions.parquet`
- `processed/v_tunnel/predictions/ensemble_predictions.parquet`
- `processed/v_tunnel/predictions/plm_ste_tiebreaker_predictions.parquet`
- `models/v_tunnel/{rf,xgb,lgbm}_multiclass.joblib`
- (optional) `processed/v_tunnel/apo_pdb_holdout.parquet`, `processed/v_tunnel/alphafold_holdout.parquet`

**Reports:**
- `reports/v_tunnel/build_summary.md` — row/structure counts, CAVER runtime, warnings breakdown, class-mean sanity table.
- `reports/v_tunnel/build_warnings.md` — every warning from the build, grouped by type and class.
- `reports/v_tunnel/plm_ste_tiebreaker_metrics.md` — winning-config headline table (auto-generated by tiebreaker module).
- `reports/v_tunnel/overall_metrics.md` — consolidated comparison (auto-generated).
- `reports/v_tunnel/feature_importance.md` — top-15 gain tables for (a) XGB multiclass on v_tunnel and (b) PLM/STE tiebreaker on v_tunnel. Mark tunnel features in **bold**.
- `reports/v_tunnel/class_mean_sanity.md` — per-class means of all 15 tunnel features.
- **Update** `reports/sterol_id_sprint.md` with a "Round 3 — tunnel features" section mirroring the Round 2 structure (bottom-line table, wins/misses, biochemistry interpretation, updated email pitch).

**Docs:**
- `docs/features/tunnel_features.md` — 3-5 paragraph human-readable description (what each feature means, why each was chosen, what class biology motivated it).

---

## 9. Time budget

Planned wall time on an 8-core laptop:

| step | wall time |
|---|---|
| CAVER install + smoke test | 15 min |
| Module implementation + unit tests | 60 min |
| Parquet build (1,780 structures × 5-15 s / 8 workers) | 30-45 min |
| Multiclass retrain (3 models × 25 iters) | 15-20 min |
| PLM/STE tiebreaker run | 2 min |
| Holdouts (optional) | 10 min |
| Report writing | 15 min |
| **Total** | **~2.5 hours** |

If any step runs long, **the extraction step is the one to parallelize harder** (bump `--workers` to match CPU count, up to 16 on the laptop). Do not skip unit tests to save time — CAVER's output format is finicky and silent parse errors are hard to debug after the fact.

---

## 10. Failure playbook

If things go sideways, here are the most likely root causes ranked by probability, with the fix for each:

1. **CAVER not finding tunnels for most pockets.**
   Root cause: starting-point coordinates in a frame CAVER doesn't accept, or probe radius too large.
   Fix: drop `probe_radius` to 0.7, bump `shell_radius` to 4.0. Re-smoke.

2. **CAVER crashing on multi-chain structures.**
   Root cause: atom renumbering breaks CAVER's Voronoi construction.
   Fix: preprocess PDBs — strip `HOH`, keep one chain per pocket (the chain closest to the pocket centroid). Cache preprocessed PDBs in `processed/v_tunnel/structures_clean/`.

3. **Row count drift after the parquet join.**
   Root cause: a structure produced 0 rows (worker crashed silently).
   Fix: always emit safe defaults for every `matched_pocket_number` in the input frame, even on failure. The ProcessPoolExecutor worker must never return fewer rows than it received.

4. **`tunnel_primary_throughput` has values > 1 or < 0.**
   Root cause: CAVER's throughput is `exp(-penalty)` and can exceed 1 in edge cases. Don't clip; document the actual range in the class-mean sanity file.

5. **Feature importance shows zero tunnel features in top-15.**
   Root cause: tunnel features are redundant with `as_max_dst`, `axial_length`, `pocket_elongation`.
   Fix: this is the kill condition. Ship a negative-result writeup, document what we learned, revert to `v_sterol` as canonical.

6. **STE F1 improves but PLM F1 regresses > 0.01.**
   Root cause: the tiebreaker is over-firing for STE.
   Fix: sweep tiebreaker margin from 0.99 back down to 0.50 in steps of 0.1; pick the margin that maximizes `5-lipid macro-F1`.

---

## 11. What this brief deliberately excludes

- **MD ensembles.** Static PDB only. If a structure has multiple NMR models, use model 1.
- **CAVER trajectory mode.** Single frame per structure.
- **Holdout recomputation of fpocket.** Holdouts already have pockets baked in (the paper's supplementary tables); we only compute tunnels on top of them.
- **New pocket detector.** This brief reuses fpocket pockets as-is.
- **Replacing the PLM/STE tiebreaker.** Keep it. Just give it better features.
- **Adding MOLE as a primary backend.** Only as an optional cross-validator on 100 structures. MOLE's main reason to exist in this brief is to flag CAVER miscalibration, not to replace CAVER.

---

## 12. Acceptance checklist (tick these before declaring done)

- [ ] CAVER 3.0.3 installed at `tools/caver/caver.jar`, version verified.
- [ ] `src/slipp_plus/tunnel_features.py` present, passes unit tests.
- [ ] `v_tunnel` registered in `FEATURE_SETS` and `FeatureSet` Literal.
- [ ] `processed/v_tunnel/full_pockets.parquet` exists, 15,219 rows, no NaN in 15 new columns.
- [ ] Class-mean sanity gates (7.2) pass.
- [ ] 3 multiclass models trained, iteration-0 joblibs saved.
- [ ] PLM/STE tiebreaker runs cleanly on v_tunnel predictions.
- [ ] `reports/v_tunnel/plm_ste_tiebreaker_metrics.md` written with headline table.
- [ ] Metric gates (7.5) evaluated and recorded in the Round 3 section of `reports/sterol_id_sprint.md`.
- [ ] Final recommendation (ship `v_tunnel` as canonical / keep `v_sterol`) documented with numeric justification.
