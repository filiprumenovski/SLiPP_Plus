#!/usr/bin/env python3
"""Export confusion matrix and per-class F1 bar chart for outreach emails.

Reads post-processed test predictions (default: promoted ste_rescue + OLA/PLM pair).
Writes PNGs under reports/v_sterol/email_attachments/ by default.
"""

from __future__ import annotations

import argparse

# Repo root on PYTHONPATH when run as module; allow script execution too
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import confusion_matrix, f1_score

_ROOT = Path(__file__).resolve().parents[1]
if str(_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(_ROOT / "src"))

from slipp_plus.constants import CLASS_10  # noqa: E402


def plot_confusion(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> Path:
    labels = np.arange(len(CLASS_10))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True).clip(min=1)

    fig, ax = plt.subplots(figsize=(8, 7))
    sns.heatmap(
        cm_norm,
        annot=cm,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_10,
        yticklabels=CLASS_10,
        cbar_kws={"label": "row-normalized rate"},
        ax=ax,
    )
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(title)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def plot_per_class_f1(y_true: np.ndarray, y_pred: np.ndarray, out: Path, title: str) -> Path:
    labels = np.arange(len(CLASS_10))
    f1 = f1_score(y_true, y_pred, labels=labels, average=None, zero_division=0)
    fig, ax = plt.subplots(figsize=(9, 4.5))
    colors = [
        "tab:blue" if c in {"CLR", "MYR", "OLA", "PLM", "STE"} else "tab:gray" for c in CLASS_10
    ]
    ax.bar(CLASS_10, f1, color=colors)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("F1 (per class)")
    ax.set_xlabel("class")
    ax.set_title(title)
    for i, v in enumerate(f1):
        ax.text(i, min(v + 0.02, 1.0), f"{v:.2f}", ha="center", fontsize=8)
    fig.tight_layout()
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--predictions",
        type=Path,
        default=Path("processed/v_sterol/predictions/ste_rescue_ola_plm_pair_predictions.parquet"),
        help="Parquet with y_true_int, y_pred_int, iteration",
    )
    p.add_argument(
        "--out-dir",
        type=Path,
        default=Path("reports/v_sterol/email_attachments"),
        help="Directory for PNG outputs",
    )
    p.add_argument("--iteration", type=int, default=0, help="Split iteration to plot")
    args = p.parse_args()

    df = pd.read_parquet(args.predictions)
    sub = df[df["iteration"] == args.iteration]
    if sub.empty:
        raise SystemExit(f"no rows for iteration={args.iteration}")

    y_true = sub["y_true_int"].to_numpy(dtype=np.int64)
    y_pred = sub["y_pred_int"].to_numpy(dtype=np.int64)
    stem = args.predictions.stem

    cm_path = args.out_dir / f"{stem}_confusion_iter{args.iteration}.png"
    f1_path = args.out_dir / f"{stem}_per_class_f1_iter{args.iteration}.png"

    plot_confusion(
        y_true,
        y_pred,
        cm_path,
        title=f"10-class confusion (iter {args.iteration}, {stem})",
    )
    plot_per_class_f1(
        y_true,
        y_pred,
        f1_path,
        title=f"Per-class F1 (iter {args.iteration}, {stem})",
    )
    print(cm_path.resolve())
    print(f1_path.resolve())


if __name__ == "__main__":
    main()
