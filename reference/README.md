# Published SLiPP reference (Dr. Dassama lab)

Official code and training table live in the public repository:

**[https://github.com/dassamalab/SLiPP_2024](https://github.com/dassamalab/SLiPP_2024)**

The **preprint PDF** (same work as the journal article; use for methods, figures, and **Table 1** benchmark numbers):

**[https://doi.org/10.1101/2024.01.26.577452](https://doi.org/10.1101/2024.01.26.577452)** → on the bioRxiv page, use **Download** / **PDF** for the full manuscript.

Peer-reviewed version: Chou et al., *J. Chem. Inf. Model.* **2024** (same binary task and dataset narrative).

The GitHub release scores pockets with a **binary** lipid-binding probability (repository README: the `prediction` column is that probability). There is **no** published multiclass lipid-type head there—only the binary task.

## Layout expected by this repo

Configs point at:

```
reference/SLiPP_2024-main/training_pockets.csv
```

First-time setup from the repository root:

```bash
mkdir -p reference
git clone https://github.com/dassamalab/SLiPP_2024 reference/SLiPP_2024-main
shasum -a 256 reference/SLiPP_2024-main/training_pockets.csv
wc -l reference/SLiPP_2024-main/training_pockets.csv
```

Expected checksum and row count for the checked-in Day 1 reference table:

```text
4d27636b4381dc3c1b9e27451db5b788e6b16f13919c4ed36f8c2ba108097711  reference/SLiPP_2024-main/training_pockets.csv
15220 reference/SLiPP_2024-main/training_pockets.csv
```

The line count includes one header row plus the 15,219 training rows enforced by the Rule 1 ingestion gate.

If you already have the upstream repository elsewhere, restore the file by either:

1. Clone into `reference/SLiPP_2024-main/` so `training_pockets.csv` sits at that path (GitHub hosts it at the repo root; renaming the clone folder to `SLiPP_2024-main` is fine), or  
2. Clone anywhere and **copy** `training_pockets.csv` into `reference/SLiPP_2024-main/`.

The companion scripts `slipp.py` and `slipp_utils.py` in that repository are the published inference pipeline; SLiPP++ is a separate codebase that consumes the **same** `training_pockets.csv` rows for fair comparison.

## What we compare against

- **Binary F1 / AUROC / holdouts:** values in `configs/*.yaml` under `ground_truth` match **Table 1** in the preprint/JCIM paper (test split + apo-PDB + AlphaFold rows). Cross-check those cells in the **preprint PDF** above or the final journal PDF.  
- **Multiclass metrics:** new in SLiPP++; there is no published multiclass baseline in [dassamalab/SLiPP_2024](https://github.com/dassamalab/SLiPP_2024).
