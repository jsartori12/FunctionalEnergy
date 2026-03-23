#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dms_plots.py
────────────
Updated version with improved aspect ratio handling for long sequences.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import matplotlib.patches as mpatches
from matplotlib.patches import Patch
from scipy.stats import gaussian_kde

log = logging.getLogger(__name__)

AA_ORDER       = list("ACDEFGHIKLMNPQRSTVWY")
CMAP_DIV       = "RdBu_r"
_DENSE_THRESH  = 40
_TICK_STEP     = 10

# Highlight style
HL_COLOR       = "#2ec4b6"   # teal
HL_EDGE        = "#0a3d62"   # dark navy border
HL_STAR        = "★"


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _save(fig, path):
    if path:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=150, bbox_inches="tight")
        log.info("Saved → %s", path)
    else:
        plt.tight_layout()
        plt.show()
    plt.close(fig)


def _pivot(df, index_col, col_col, value_col):
    pt = df.pivot_table(index=index_col, columns=col_col,
                        values=value_col, aggfunc="mean")
    return pt[[aa for aa in AA_ORDER if aa in pt.columns]]


def _sym_lim(arr, pct=95):
    return float(np.nanpercentile(np.abs(arr), pct))


def _xticks(ax, positions, wt_map, highlight=None):
    """
    Smart x-axis labels.
    Highlighted positions get a coloured label and ★ prefix.
    """
    n  = len(positions)
    hl = set(highlight or [])

    if n <= _DENSE_THRESH:
        idxs = list(range(n))
        labels = []
        for i in idxs:
            pos = positions[i]
            wt  = wt_map.get(pos, "")
            txt = f"{HL_STAR}{pos}\n{wt}" if pos in hl else f"{pos}\n{wt}"
            labels.append(txt)
        rot, ha, fs = 0, "center", 9
    else:
        idxs   = list(range(0, n, _TICK_STEP))
        labels = []
        for i in idxs:
            pos = positions[i]
            txt = f"{HL_STAR}{pos}" if pos in hl else str(pos)
            labels.append(txt)
        rot, ha, fs = 45, "right", 8

    ax.set_xticks(idxs)
    ticklabels = ax.set_xticklabels(labels, rotation=rot, ha=ha, fontsize=fs)

    if hl and n <= _DENSE_THRESH:
        for i, tl in zip(range(n), ticklabels):
            if positions[i] in hl:
                tl.set_color(HL_EDGE)
                tl.set_fontweight("bold")

    suffix = " · WT below" if n <= _DENSE_THRESH else f" · every {_TICK_STEP} shown"
    ax.set_xlabel(f"Residue index{suffix}", fontsize=10)


def _wt_dots(ax, pivot, wt_map):
    """Black dot on the WT position in heatmaps."""
    cols = pivot.columns.tolist()
    for xi, pos in enumerate(pivot.index):
        wt = wt_map.get(pos)
        if wt and wt in cols:
            ax.plot(xi, cols.index(wt), "k.", markersize=5, alpha=0.75)


def _highlight_heatmap_cols(ax, pivot, highlight):
    """Draws a coloured rectangle around highlighted columns."""
    if not highlight:
        return
    hl    = set(highlight)
    n_aa  = len(pivot.columns)
    for xi, pos in enumerate(pivot.index):
        if pos in hl:
            rect = mpatches.Rectangle(
                (xi - 0.5, -0.5), 1, n_aa,
                linewidth=2, edgecolor=HL_EDGE,
                facecolor=HL_COLOR, alpha=0.15, zorder=3,
            )
            ax.add_patch(rect)
            ax.text(xi, -1.0, HL_STAR, ha="center", va="top",
                    color=HL_EDGE, fontsize=11, fontweight="bold",
                    clip_on=False)


def _highlight_bars(ax, positions, values, highlight, label="Catalytic"):
    """Adds hatch pattern and a top label to highlighted bars."""
    if not highlight:
        return None
    hl = set(highlight)
    added = False
    for xi, pos in enumerate(positions):
        if pos in hl:
            ax.bar(xi, values[xi],
                   color="none", edgecolor=HL_EDGE,
                   linewidth=2, hatch="///", width=0.85, zorder=4)
            ax.text(xi, values[xi], HL_STAR,
                    ha="center", va="bottom",
                    color=HL_EDGE, fontsize=12, fontweight="bold")
            added = True
    if added:
        return Patch(facecolor="none", edgecolor=HL_EDGE,
                     hatch="///", linewidth=1.5, label=label)
    return None

def _get_heatmap_dims(n_pos):
    """Calculates balanced figure dimensions and gridspec ratios."""
    # Scale width more conservatively to avoid extreme aspect ratios
    # For ~270 pos, this gives ~25-30 inches instead of 100+
    fig_w = max(15, n_pos * 0.12 + 2.0)
    fig_h = 7.5 # Slightly taller for better readability

    # Adjust width ratio for colorbar based on total width
    cbar_w = 0.4 if fig_w < 25 else 0.6
    return fig_w, fig_h, [fig_w - cbar_w, cbar_w]

# ============================================================================
# 1. Rosetta DMS
# ============================================================================

def plot_rosetta_dms(df, output_prefix=None, energy_col="ddG_total_energy",
                     vmin=None, vmax=None, highlight_positions=None):
    if energy_col not in df.columns:
        raise ValueError(f"Column '{energy_col}' not found.")

    pivot     = _pivot(df, "Position_Pose", "Mutation", energy_col)
    positions = list(pivot.index)
    wt_map    = (df.drop_duplicates("Position_Pose")
                   .set_index("Position_Pose")["WT"].to_dict()
                 if "WT" in df.columns else {})

    lim  = _sym_lim(pivot.values) if vmin is None else max(abs(vmin), abs(vmax))
    vmin, vmax = -lim, lim

    # Heatmap
    fw, fh, wr = _get_heatmap_dims(len(positions))
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh), gridspec_kw={"width_ratios": wr, "wspace": 0.03})
    ax, cax = axes

    im = ax.imshow(pivot.T.values, aspect="auto", cmap=CMAP_DIV, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, highlight_positions)

    ax.set_title("Rosetta DMS — ΔΔG per variant", fontsize=13, fontweight="bold")
    fig.colorbar(im, cax=cax).set_label("ΔΔG (kcal/mol)")
    _save(fig, f"{output_prefix}_rosetta_heatmap.png" if output_prefix else None)

    # Barplot
    site_mean = df.groupby("Position_Pose")[energy_col].mean().reindex(positions)
    fig, ax   = plt.subplots(figsize=(fw, 4.5))
    colors    = ["#e63946" if v > 0 else "#457b9d" for v in site_mean]
    ax.bar(range(len(positions)), site_mean.values, color=colors, width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _save(fig, f"{output_prefix}_rosetta_barplot.png" if output_prefix else None)


# ============================================================================
# 2. ESM-2 evolutionary scores
# ============================================================================

def plot_evo_scores(df, output_prefix=None, vmin=None, vmax=None,
                    highlight_positions=None):
    if "delta_psi_evo" not in df.columns:
        raise ValueError("Column 'delta_psi_evo' not found.")

    pivot     = _pivot(df, "Position_1based", "Mutation", "delta_psi_evo")
    positions = list(pivot.index)
    wt_map    = (df.drop_duplicates("Position_1based")
                   .set_index("Position_1based")["WT"].to_dict()
                 if "WT" in df.columns else {})

    lim  = _sym_lim(pivot.values) if vmin is None else max(abs(vmin), abs(vmax))
    vmin, vmax = -lim, lim

    # Heatmap
    fw, fh, wr = _get_heatmap_dims(len(positions))
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh), gridspec_kw={"width_ratios": wr, "wspace": 0.03})
    ax, cax = axes

    im = ax.imshow(pivot.T.values, aspect="auto", cmap=CMAP_DIV, vmin=vmin, vmax=vmax)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, highlight_positions)

    ax.set_title("ESM-2 — ΔΨᵉᵛᵒ per variant", fontsize=13, fontweight="bold")
    fig.colorbar(im, cax=cax).set_label("ΔΨᵉᵛᵒ (higher = more deleterious)")
    _save(fig, f"{output_prefix}_evo_heatmap.png" if output_prefix else None)

    # Per-AA mean
    aa_mean = (df.groupby("Mutation")["delta_psi_evo"]
                 .mean()
                 .reindex([a for a in AA_ORDER if a in df["Mutation"].unique()]))
    fig, ax = plt.subplots(figsize=(5, 7))
    colors  = ["#e63946" if v > 0 else "#457b9d" for v in aa_mean.values]
    ax.barh(range(len(aa_mean)), aa_mean.values, color=colors)
    ax.set_yticks(range(len(aa_mean)))
    ax.set_yticklabels(aa_mean.index)
    ax.set_title("ESM-2 — Mean ΔΨᵉᵛᵒ per mutant AA", fontweight="bold")
    _save(fig, f"{output_prefix}_evo_aa_mean.png" if output_prefix else None)


# ============================================================================
# 3. Dark energy
# ============================================================================

def plot_dark_energy(df, output_prefix=None, threshold=None,
                     highlight_positions=None):
    required = {"ddG_total_energy", "delta_psi_evo_scaled",
                "dark_energy", "Position_1based", "WT", "Mutation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if threshold is None:
        threshold = float(np.percentile(df["dark_energy"], 75))

    de_vals = df["dark_energy"].values
    vabs    = _sym_lim(de_vals)
    norm    = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    pivot     = _pivot(df, "Position_1based", "Mutation", "dark_energy")
    positions = list(pivot.index)
    wt_map    = df.drop_duplicates("Position_1based").set_index("Position_1based")["WT"].to_dict()

    # Landscape Scatter
    fig, ax = plt.subplots(figsize=(7, 6))
    sc = ax.scatter(df["ddG_total_energy"], df["delta_psi_evo_scaled"],
                    c=df["dark_energy"], cmap=CMAP_DIV, norm=norm, s=15, alpha=0.6)
    ax.set_xlabel("ΔEfold (Rosetta ΔΔG)")
    ax.set_ylabel("kBT_sel · ΔΨᵉᵛᵒ")
    fig.colorbar(sc, ax=ax).set_label("ΔEᵈᵃʳᵏ (kcal/mol)")
    _save(fig, f"{output_prefix}_de_landscape.png" if output_prefix else None)

    # Heatmap
    fw, fh, wr = _get_heatmap_dims(len(positions))
    fig, axes = plt.subplots(1, 2, figsize=(fw, fh), gridspec_kw={"width_ratios": wr, "wspace": 0.03})
    ax, cax = axes
    im = ax.imshow(pivot.T.values, aspect="auto", cmap=CMAP_DIV, norm=norm)
    ax.set_yticks(range(len(pivot.columns)))
    ax.set_yticklabels(pivot.columns)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, highlight_positions)
    fig.colorbar(im, cax=cax).set_label("ΔEᵈᵃʳᵏ (kcal/mol)")
    _save(fig, f"{output_prefix}_de_heatmap.png" if output_prefix else None)

    # Site-average
    site_avg = df.groupby("Position_1based")["dark_energy"].mean().reindex(positions)
    fig, ax = plt.subplots(figsize=(fw, 4.5))
    colors = ["#e63946" if v >= threshold else "#457b9d" for v in site_avg]
    ax.bar(range(len(positions)), site_avg.values, color=colors)
    ax.axhline(threshold, color="#f4a261", ls="--", label="Threshold")
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _save(fig, f"{output_prefix}_de_site_avg.png" if output_prefix else None)

# Convenience wrapper and CLI remain the same...
def plot_all(rosetta_csv=None, esm_csv=None, dark_csv=None,
             output_prefix=None, threshold=None, highlight_positions=None):
    if rosetta_csv:
        plot_rosetta_dms(pd.read_csv(rosetta_csv), output_prefix=output_prefix,
                         highlight_positions=highlight_positions)
    if esm_csv:
        plot_evo_scores(pd.read_csv(esm_csv), output_prefix=output_prefix,
                        highlight_positions=highlight_positions)
    if dark_csv:
        plot_dark_energy(pd.read_csv(dark_csv), output_prefix=output_prefix,
                         threshold=threshold,
                         highlight_positions=highlight_positions)

if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--rosetta-csv",   default=None)
    p.add_argument("--esm-csv",       default=None)
    p.add_argument("--dark-csv",      default=None)
    p.add_argument("--output-prefix", default=None)
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--highlight-positions", nargs="+", type=int, default=None)
    args = p.parse_args()
    plot_all(args.rosetta_csv, args.esm_csv, args.dark_csv,
             args.output_prefix, args.threshold, args.highlight_positions)
