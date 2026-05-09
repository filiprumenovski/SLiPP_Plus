"""Publication-grade figure rendering for SLiPP++.

This module produces a publication-ready set of figures, designed as a direct
upgrade over Chou et al. 2024's figures. The headline figure (``figure_7_plus``)
is a four-panel feature-importance + per-class separability figure that:

- breaks the lipid bucket into all five sub-classes (CLR/MYR/OLA/PLM/STE),
  rather than collapsing them as the original figure does,
- shows feature-family-level importance alongside per-feature importance, so
  the contribution of the new SLiPP++ feature stacks (aa20, shell12,
  sterol_chemistry, tunnel_shape) is visible,
- replaces matplotlib defaults with a Tol-inspired class-aware palette and
  tighter typography,
- adds density contours to the PCA panel so per-class manifold structure is
  legible rather than buried in scatter overplotting.

Each figure renderer accepts plain Python data structures (pandas DataFrames,
numpy arrays, dicts) so it can be tested against synthetic fixtures and run
against the real prediction / model artifacts when they are available.

Output: SVG (editorial), PDF (journal), and 300-DPI PNG (slides) per figure,
written under ``figures/`` at the repository root.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .constants import CLASS_10, LIPID_CODES

# ----------------------------------------------------------------------
# Style system
# ----------------------------------------------------------------------

#: Class-aware palette. Lipid sub-classes are warm hues that sort visually as
#: a chemistry gradient (sterol family vs. fatty-acid family). Non-lipid
#: classes use cool / neutral hues so the lipid story dominates the figure.
CLASS_PALETTE: dict[str, str] = {
    # Lipids — warm
    "CLR": "#B0413E",  # cholesterol — deep red
    "MYR": "#E69138",  # myristic acid — orange
    "OLA": "#D55E00",  # oleic acid — burnt orange
    "PLM": "#8E5524",  # palmitic acid — brown
    "STE": "#A23B72",  # stearic acid — wine
    # Non-lipids — cool
    "ADN": "#2E86AB",  # adenosine — blue
    "B12": "#0F8B8D",  # cobalamin — teal
    "BGC": "#3E8E41",  # beta-glucose — green
    "COA": "#5E548E",  # coenzyme A — purple
    "PP": "#9AA0A6",   # pseudo-pocket — neutral gray
}

#: Feature-family palette used in importance plots and the schematic.
FAMILY_PALETTE: dict[str, str] = {
    "paper17": "#34495E",
    "aa20": "#16A085",
    "shell12": "#9B59B6",
    "sterol_chemistry": "#E67E22",
    "tunnel_shape": "#C0392B",
    "derived": "#7F8C8D",
    "boundary22": "#2C3E50",
}

#: Output formats produced for every figure (vector + raster).
DEFAULT_FORMATS: tuple[str, ...] = ("png", "pdf", "svg")


def apply_publication_style() -> None:
    """Apply the project's publication rcParams.

    Tight, modern, journal-leaning. Designed to look right at column-width
    (~3.5 in) and full-page (~7 in) without further tuning.
    """

    plt.rcParams.update(
        {
            "figure.dpi": 110,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.08,
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#444444",
            "axes.linewidth": 0.8,
            "axes.labelcolor": "#222222",
            "axes.titlesize": 12,
            "axes.titleweight": "semibold",
            "axes.titlepad": 8.0,
            "axes.labelsize": 10.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "grid.color": "#E5E5E5",
            "grid.linewidth": 0.6,
            "xtick.color": "#444444",
            "ytick.color": "#444444",
            "xtick.labelsize": 9.5,
            "ytick.labelsize": 9.5,
            "legend.frameon": False,
            "legend.fontsize": 9.0,
            "legend.title_fontsize": 9.5,
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans", "Arial", "Helvetica"],
            "font.size": 10.0,
            "pdf.fonttype": 42,  # editable text in vector output
            "ps.fonttype": 42,
        }
    )


def save_figure(fig: plt.Figure, out_dir: Path, stem: str, formats: Sequence[str] = DEFAULT_FORMATS) -> dict[str, Path]:
    """Save ``fig`` as every format in ``formats`` and return the path map."""

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for fmt in formats:
        path = out_dir / f"{stem}.{fmt}"
        fig.savefig(path, format=fmt)
        paths[fmt] = path
    return paths


def set_title_block(
    fig: plt.Figure,
    title: str,
    subtitle: str,
    *,
    x: float = 0.06,
    title_size: float = 15.0,
    subtitle_size: float = 10.5,
) -> None:
    """Apply the standard two-line publication title block.

    The title and subtitle are positioned with enough vertical breathing room
    that the title's descenders never touch the subtitle's ascenders, even at
    print resolution. Callers are responsible for `subplots_adjust(top=...)`
    so the panel headers don't crowd the subtitle from below.
    """

    fig.suptitle(title, x=x, y=0.985, ha="left", va="top", fontsize=title_size, fontweight="bold")
    fig.text(x, 0.935, subtitle, ha="left", va="top", fontsize=subtitle_size, color="#555555")


# ----------------------------------------------------------------------
# Inputs
# ----------------------------------------------------------------------


@dataclass
class FeatureImportanceData:
    """Inputs for the feature-importance panels of the headline figure."""

    #: Top-N feature names, ordered by overall importance.
    features: list[str]
    #: Feature-family label per feature (must match :data:`FAMILY_PALETTE` keys).
    families: list[str]
    #: Mean decrease in impurity (Gini), one value per feature.
    mdi: np.ndarray
    #: Mean decrease in F1 from permutation, one value per feature.
    permutation_mean: np.ndarray
    #: Standard deviation of permutation decrease across CV iterations.
    permutation_std: np.ndarray


@dataclass
class PerClassFeatureDistribution:
    """Inputs for the per-class violin panel."""

    #: Feature being plotted (e.g. ``"hydrophobicity_score"`` or ``"apol_as_prop"``).
    feature_name: str
    #: One array of feature values per class label, keyed by CLASS_10 entries.
    values_by_class: dict[str, np.ndarray]


@dataclass
class EmbeddingData:
    """Inputs for the 2D embedding panel."""

    #: 2D coordinates for every pocket, shape ``(n_rows, 2)``.
    coords: np.ndarray
    #: Class label per pocket, length ``n_rows``.
    classes: np.ndarray
    #: Variance-explained ratio for the two components, e.g. ``(0.41, 0.18)``.
    explained_variance_ratio: tuple[float, float]
    #: Embedding label, used for axis titles (``"PCA"`` or ``"UMAP"``).
    method: str = "PCA"


@dataclass
class ExperimentSummary:
    """One row of summary metrics for a registry experiment."""

    exp_id: str
    label: str
    lipid5_macro_f1: float
    apo_pdb_f1: float | None
    alphafold_f1: float | None
    is_deployable: bool = False
    is_internal_best: bool = False
    is_paper_baseline: bool = False
    supersedes: str | None = None


@dataclass
class PerClassResult:
    """Per-class F1 with optional uncertainty for one experiment."""

    exp_id: str
    label: str
    per_class_f1: dict[str, float]
    per_class_f1_std: dict[str, float] = field(default_factory=dict)


# ----------------------------------------------------------------------
# Headline figure: a better Figure 7
# ----------------------------------------------------------------------


def _format_class_axis(ax: plt.Axes, class_order: Sequence[str]) -> None:
    """Apply the project's standard class-axis styling (tick colors, etc.)."""

    ax.set_xticks(np.arange(len(class_order)))
    ax.set_xticklabels(
        class_order,
        rotation=0,
        fontsize=9.5,
    )
    for tick, cls in zip(ax.get_xticklabels(), class_order, strict=True):
        tick.set_color(CLASS_PALETTE.get(cls, "#333333"))
        tick.set_fontweight("semibold")


def figure_7_plus(
    *,
    importance: FeatureImportanceData,
    distribution: PerClassFeatureDistribution,
    embedding: EmbeddingData,
    out_dir: Path,
    stem: str = "figure7_plus_feature_landscape",
    formats: Sequence[str] = DEFAULT_FORMATS,
    title: str = "Pocket physicochemistry organizes ten ligand classes",
    subtitle: str = "Extending Chou et al. 2024 Fig. 7 with the SLiPP++ ten-class softmax and richer feature families",
) -> dict[str, Path]:
    """Render the headline four-panel feature-landscape figure.

    Layout (same logical roles as Chou et al. Fig. 7, broken into the
    project's richer ten-class taxonomy):

    - Panel A: feature-family importance (mean decrease in impurity, summed
      within family).
    - Panel B: top-N individual features by permutation F1 decrease, with
      one-standard-deviation error bars across the CV iterations.
    - Panel C: per-class violin plots of the most discriminating feature, with
      lipid sub-classes shown side-by-side rather than collapsed.
    - Panel D: 2D embedding (PCA or UMAP) of all pockets with per-class
      density contours.
    """

    apply_publication_style()
    fig = plt.figure(figsize=(14.0, 11.5))
    gs = fig.add_gridspec(2, 2, hspace=0.50, wspace=0.30, left=0.06, right=0.985, top=0.86, bottom=0.06)
    ax_a = fig.add_subplot(gs[0, 0])
    ax_b = fig.add_subplot(gs[1, 0])
    ax_c = fig.add_subplot(gs[0, 1])
    ax_d = fig.add_subplot(gs[1, 1])

    set_title_block(fig, title, subtitle, title_size=15.5)

    _draw_panel_a_family_importance(ax_a, importance)
    _draw_panel_b_top_features(ax_b, importance)
    _draw_panel_c_per_class_violin(ax_c, distribution)
    _draw_panel_d_embedding(ax_d, embedding)

    _label_panel(ax_a, "A")
    _label_panel(ax_b, "B")
    _label_panel(ax_c, "C")
    _label_panel(ax_d, "D")

    return save_figure(fig, out_dir, stem, formats)


def _label_panel(ax: plt.Axes, letter: str) -> None:
    ax.text(
        -0.085,
        1.04,
        letter,
        transform=ax.transAxes,
        fontsize=15,
        fontweight="bold",
        ha="left",
        va="bottom",
    )


def _draw_panel_a_family_importance(ax: plt.Axes, importance: FeatureImportanceData) -> None:
    """Family-aggregated MDI as a horizontal bar chart."""

    df = pd.DataFrame(
        {
            "feature": importance.features,
            "family": importance.families,
            "mdi": importance.mdi,
        }
    )
    by_family = (
        df.groupby("family", as_index=False)["mdi"].sum().sort_values("mdi", ascending=True)
    )
    ax.barh(
        by_family["family"],
        by_family["mdi"],
        color=[FAMILY_PALETTE.get(f, "#888888") for f in by_family["family"]],
        edgecolor="white",
        linewidth=0.8,
    )
    for fam, value in zip(by_family["family"], by_family["mdi"], strict=True):
        ax.text(
            value + max(by_family["mdi"]) * 0.012,
            fam,
            f"{value:.3f}",
            va="center",
            fontsize=9,
            color="#444444",
        )
    ax.set_title("Feature-family importance (mean decrease in impurity)", loc="left")
    ax.set_xlabel("Cumulative decrease in impurity")
    ax.set_xlim(0, by_family["mdi"].max() * 1.18)
    ax.tick_params(axis="y", which="both", length=0)
    for label in ax.get_yticklabels():
        label.set_fontweight("semibold")


def _draw_panel_b_top_features(ax: plt.Axes, importance: FeatureImportanceData, top_n: int = 15) -> None:
    """Top-N per-feature permutation importance with std error bars."""

    df = pd.DataFrame(
        {
            "feature": importance.features,
            "family": importance.families,
            "perm_mean": importance.permutation_mean,
            "perm_std": importance.permutation_std,
        }
    ).sort_values("perm_mean", ascending=False).head(top_n).reset_index(drop=True)

    xs = np.arange(len(df))
    colors = [FAMILY_PALETTE.get(f, "#888888") for f in df["family"]]
    ax.bar(
        xs,
        df["perm_mean"],
        yerr=df["perm_std"],
        color=colors,
        edgecolor="white",
        linewidth=0.6,
        capsize=2.5,
        error_kw={"elinewidth": 0.8, "ecolor": "#333333"},
    )
    ax.set_xticks(xs)
    ax.set_xticklabels(df["feature"], rotation=30, ha="right", fontsize=8.5)
    ax.set_ylabel("Permutation decrease in macro-F1")
    ax.set_title("Top features by permutation importance (25-iter CV, 1σ)", loc="left")
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    handles = [
        mpatches.Patch(color=FAMILY_PALETTE[f], label=f)
        for f in pd.unique(df["family"])
        if f in FAMILY_PALETTE
    ]
    ax.legend(handles=handles, loc="upper right", title="feature family", ncol=1)


def _draw_panel_c_per_class_violin(ax: plt.Axes, dist: PerClassFeatureDistribution) -> None:
    """Per-class violin: lipids on the left (warm), non-lipids on the right (cool)."""

    lipid_order = [c for c in CLASS_10 if c in LIPID_CODES]
    nonlipid_order = [c for c in CLASS_10 if c not in LIPID_CODES]
    class_order = lipid_order + nonlipid_order

    data = [np.asarray(dist.values_by_class.get(c, np.array([])), dtype=np.float64) for c in class_order]
    positions = np.arange(len(class_order))
    parts = ax.violinplot(
        data,
        positions=positions,
        widths=0.78,
        showmedians=False,
        showextrema=False,
    )
    for body, cls in zip(parts["bodies"], class_order, strict=True):
        body.set_facecolor(CLASS_PALETTE[cls])
        body.set_edgecolor("white")
        body.set_alpha(0.85)
    # Median + IQR overlay for legibility
    for pos, arr in zip(positions, data, strict=True):
        if arr.size == 0:
            continue
        q1, med, q3 = np.percentile(arr, [25, 50, 75])
        ax.plot([pos, pos], [q1, q3], color="#222222", linewidth=2.2, solid_capstyle="round")
        ax.plot([pos], [med], marker="o", markersize=4.5, color="white", markeredgecolor="#222222", markeredgewidth=0.8)
        ax.text(pos, ax.get_ylim()[0] if ax.get_ylim()[0] != 0 else min(arr) - 0.05 * (max(arr) - min(arr) + 1e-6), f"n={arr.size}", ha="center", va="top", fontsize=8, color="#666666")

    # Vertical separator between lipid and non-lipid groups
    sep = len(lipid_order) - 0.5
    ax.axvline(sep, color="#BBBBBB", linewidth=0.8, linestyle="--", alpha=0.7)
    ax.text(
        len(lipid_order) / 2 - 0.5,
        ax.get_ylim()[1] if any(d.size for d in data) else 1.0,
        "lipid",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#A23B72",
        fontweight="semibold",
    )
    ax.text(
        len(lipid_order) + len(nonlipid_order) / 2 - 0.5,
        ax.get_ylim()[1] if any(d.size for d in data) else 1.0,
        "non-lipid",
        ha="center",
        va="bottom",
        fontsize=9.5,
        color="#2E86AB",
        fontweight="semibold",
    )

    _format_class_axis(ax, class_order)
    ax.set_ylabel(dist.feature_name.replace("_", " "))
    ax.set_title(f"Per-class distribution of {dist.feature_name} — lipid sub-classes are not interchangeable", loc="left")


def _draw_panel_d_embedding(ax: plt.Axes, emb: EmbeddingData) -> None:
    """2D embedding scatter with per-class density contours."""

    classes = np.asarray(emb.classes)
    coords = np.asarray(emb.coords)
    # Background: pseudo-pocket scatter (gray, low alpha) so it doesn't dominate
    pp_mask = classes == "PP"
    if pp_mask.any():
        ax.scatter(
            coords[pp_mask, 0],
            coords[pp_mask, 1],
            s=4,
            color=CLASS_PALETTE["PP"],
            alpha=0.18,
            linewidths=0,
            zorder=1,
        )
    # Foreground: every other class, colored, with density contours over the
    # densest classes (lipids especially) to show structure not legible from
    # raw scatter overplotting.
    for cls in CLASS_10:
        if cls == "PP":
            continue
        mask = classes == cls
        if mask.sum() < 5:
            continue
        ax.scatter(
            coords[mask, 0],
            coords[mask, 1],
            s=10 if cls in LIPID_CODES else 7,
            color=CLASS_PALETTE[cls],
            alpha=0.55 if cls in LIPID_CODES else 0.4,
            linewidths=0,
            zorder=3 if cls in LIPID_CODES else 2,
            label=cls,
        )

    # Contours for the lipid classes: where they cluster, with the class color.
    for cls in [c for c in CLASS_10 if c in LIPID_CODES]:
        mask = classes == cls
        if mask.sum() < 30:
            continue
        try:
            from scipy.stats import gaussian_kde

            xy = coords[mask].T
            kde = gaussian_kde(xy)
            xmin, xmax = ax.get_xlim()
            ymin, ymax = ax.get_ylim()
            xi, yi = np.mgrid[xmin:xmax:80j, ymin:ymax:80j]
            zi = kde(np.vstack([xi.flatten(), yi.flatten()])).reshape(xi.shape)
            levels = np.linspace(zi.max() * 0.25, zi.max() * 0.95, 4)
            ax.contour(
                xi,
                yi,
                zi,
                levels=levels,
                colors=[CLASS_PALETTE[cls]],
                linewidths=0.7,
                alpha=0.75,
                zorder=4,
            )
        except Exception:
            # KDE can fail when data is degenerate; fall back to scatter only.
            pass

    pc1 = emb.explained_variance_ratio[0] * 100
    pc2 = emb.explained_variance_ratio[1] * 100
    ax.set_xlabel(f"{emb.method} 1 ({pc1:.1f}%)")
    ax.set_ylabel(f"{emb.method} 2 ({pc2:.1f}%)")
    ax.set_title(f"{emb.method} of pocket descriptors — per-class density contours over scatter", loc="left")
    # Place the legend outside the axes (right side) so it never overlaps data.
    leg = ax.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        ncol=1,
        fontsize=8.5,
        title="class",
        title_fontsize=9.0,
        markerscale=1.6,
        labelspacing=0.45,
        handletextpad=0.4,
    )
    leg.get_frame().set_alpha(0.0)


# ----------------------------------------------------------------------
# Per-class forest plot
# ----------------------------------------------------------------------


def figure_per_class_forest(
    *,
    results: list[PerClassResult],
    paper_baseline_per_class: dict[str, float] | None = None,
    out_dir: Path,
    stem: str = "figure_per_class_forest",
    formats: Sequence[str] = DEFAULT_FORMATS,
    title: str = "Per-class F1 across the SLiPP++ experiment ladder",
    subtitle: str = "Five lipid sub-classes per experiment; vertical dashed lines mark the Chou et al. 2024 paper baseline for the same class.",
) -> dict[str, Path]:
    """Forest plot of per-class F1 across selected experiments.

    One row per experiment, dots for each class colored by the project palette,
    optional vertical reference line for the paper baseline.
    """

    apply_publication_style()
    # Lipid-only forest: drop any experiment row that doesn't have all five
    # lipid sub-classes recorded, so the plot never has visual gaps.
    lipid_classes = [c for c in CLASS_10 if c in LIPID_CODES]
    complete = [r for r in results if all(c in r.per_class_f1 for c in lipid_classes)]
    if not complete:
        complete = results  # fallback: render whatever we have
    n_exps = len(complete)
    fig_height = max(4.0, 0.75 * n_exps + 2.6)
    fig, ax = plt.subplots(figsize=(12.5, fig_height))

    # Y axis: experiments stacked vertically, top is best (last in list)
    y_positions = np.arange(n_exps)
    for i, result in enumerate(complete):
        for j, cls in enumerate(lipid_classes):
            value = result.per_class_f1.get(cls)
            if value is None:
                continue
            std = result.per_class_f1_std.get(cls, 0.0)
            x_jitter = (j - (len(lipid_classes) - 1) / 2) * 0.012
            ax.errorbar(
                value,
                i + x_jitter * 6,
                xerr=std if std else None,
                fmt="o",
                color=CLASS_PALETTE[cls],
                ecolor=CLASS_PALETTE[cls],
                elinewidth=1.0,
                capsize=2,
                markersize=8,
                alpha=0.95,
                markeredgecolor="white",
                markeredgewidth=0.6,
                zorder=3,
            )

    # Paper baseline reference lines per class (lipid only)
    if paper_baseline_per_class:
        for cls, baseline in paper_baseline_per_class.items():
            if cls not in lipid_classes:
                continue
            ax.axvline(
                baseline,
                color=CLASS_PALETTE[cls],
                linestyle="--",
                linewidth=0.7,
                alpha=0.45,
                zorder=1,
            )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([r.label for r in complete], fontsize=10.0)
    ax.invert_yaxis()  # top = first; flip so the deployable lands at top visually
    ax.set_xlabel("Class F1")
    ax.set_xlim(0.2, 1.0)
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    set_title_block(fig, title, subtitle, title_size=14.5, subtitle_size=9.8)

    # Single lipid-only legend, centered below the plot. Non-lipid classes
    # are deliberately omitted because per-class F1 for them is not tracked
    # in the registry; including their patches would falsely imply they are
    # plotted somewhere on this figure.
    lipid_handles = [mpatches.Patch(color=CLASS_PALETTE[c], label=c) for c in lipid_classes]
    fig.legend(
        handles=lipid_handles,
        title="lipid sub-classes",
        loc="lower center",
        bbox_to_anchor=(0.5, 0.01),
        fontsize=9.5,
        ncol=len(lipid_classes),
        title_fontsize=10.0,
    )

    plt.subplots_adjust(left=0.32, right=0.985, top=0.84, bottom=0.20)
    return save_figure(fig, out_dir, stem, formats)


# ----------------------------------------------------------------------
# Ablation ladder
# ----------------------------------------------------------------------


def figure_ablation_ladder(
    *,
    rows: list[dict[str, Any]],
    out_dir: Path,
    stem: str = "figure_ablation_ladder",
    formats: Sequence[str] = DEFAULT_FORMATS,
    title: str = "Ablation ladder — feature stack and ensembling deltas",
    subtitle: str = "Sequential gains in 5-lipid macro-F1 from the paper baseline (paper17) to the SLiPP++ deployable (exp-021).",
) -> dict[str, Path]:
    """Step-style chart of the lipid5 macro-F1 trajectory across the ablation ladder.

    ``rows`` items must have keys ``label`` (str), ``lipid5`` (float),
    ``binary_f1`` (float), ``family`` (str, key of FAMILY_PALETTE) and
    optionally ``annotation`` (str).
    """

    apply_publication_style()
    fig, ax = plt.subplots(figsize=(11.5, 5.6))
    xs = np.arange(len(rows))
    ys = np.array([r["lipid5"] for r in rows])
    families = [r.get("family", "paper17") for r in rows]
    colors = [FAMILY_PALETTE.get(f, "#888888") for f in families]

    # Step line connecting consecutive points
    ax.plot(xs, ys, color="#34495E", linewidth=1.3, alpha=0.5, zorder=1)
    for i, (x, y, c, row) in enumerate(zip(xs, ys, colors, rows, strict=True)):
        ax.scatter(x, y, s=160, color=c, edgecolor="white", linewidth=1.0, zorder=3)
        ax.text(
            x,
            y + 0.012,
            f"{y:.3f}",
            ha="center",
            fontsize=9.5,
            color="#222222",
            fontweight="semibold",
        )
        if i > 0:
            delta = ys[i] - ys[i - 1]
            sign = "+" if delta >= 0 else ""
            ax.text(
                x,
                y - 0.022,
                f"{sign}{delta:.3f}",
                ha="center",
                fontsize=8.5,
                color="#16A085" if delta >= 0 else "#B0413E",
            )
        ann = row.get("annotation")
        if ann:
            ax.text(
                x,
                y - 0.045,
                ann,
                ha="center",
                fontsize=8,
                color="#555555",
                style="italic",
            )

    ax.set_xticks(xs)
    ax.set_xticklabels([r["label"] for r in rows], rotation=18, ha="right", fontsize=9.0)
    ax.set_ylabel("5-lipid macro-F1 (25-iter CV)")
    ax.set_ylim(min(ys) - 0.07, max(ys) + 0.06)
    ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.7)
    ax.set_axisbelow(True)
    set_title_block(fig, title, subtitle, title_size=14.5, subtitle_size=9.8)

    # Family legend
    used = list(pd.unique(pd.Series(families)))
    handles = [mpatches.Patch(color=FAMILY_PALETTE[f], label=f) for f in used if f in FAMILY_PALETTE]
    ax.legend(handles=handles, title="dominant feature family", loc="lower right")

    plt.subplots_adjust(left=0.075, right=0.985, top=0.85, bottom=0.20)
    return save_figure(fig, out_dir, stem, formats)
