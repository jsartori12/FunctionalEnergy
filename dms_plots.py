#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dms_plots.py
────────────

Cada função gera figuras SEPARADAS (um arquivo PNG por plot).

  plot_rosetta_dms()  →  _rosetta_heatmap.png  +  _rosetta_barplot.png
  plot_evo_scores()   →  _evo_heatmap.png       +  _evo_aa_mean.png
  plot_dark_energy()  →  _de_landscape.png  +  _de_heatmap.png
                         _de_site_avg.png   +  _de_distribution.png

Eixo X inteligente:
  ≤ 40 posições  → mostra índice + WT em cada posição
  > 40 posições  → mostra só o índice a cada 10 posições (sem poluição)

highlight_positions:
  Aceita uma lista de posições (1-based / Pose numbering) para marcar
  resíduos de interesse (ex: sítio catalítico).
  No heatmap → coluna com borda colorida + estrela no topo.
  No barplot/site-avg → barra com hatch e label acima.
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

    # colour highlighted tick labels
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
    """
    Draws a coloured rectangle around each highlighted column in a heatmap
    and places a ★ above it.
    """
    if not highlight:
        return
    hl    = set(highlight)
    n_aa  = len(pivot.columns)
    for xi, pos in enumerate(pivot.index):
        if pos in hl:
            # Rectangle around the column  (x, y, w, h) in data coords
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
    """
    Adds hatch pattern and a top label to highlighted bars.
    Returns a Patch for the legend (or None).
    """
    if not highlight:
        return None
    hl = set(highlight)
    added = False
    for xi, pos in enumerate(positions):
        if pos in hl:
            # Overlay a hatched bar of the same height
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


# ============================================================================
# 1.  Rosetta DMS  →  2 figures
# ============================================================================

def plot_rosetta_dms(df, output_prefix=None, energy_col="ddG_total_energy",
                     vmin=None, vmax=None, highlight_positions=None):
    """
    Parameters
    ----------
    highlight_positions : list of int, optional
        Pose-numbered positions to mark as catalytic / functional sites.
        E.g.: highlight_positions=[70, 73, 130]
    """
    if energy_col not in df.columns:
        raise ValueError(f"Column '{energy_col}' not found.")

    pivot     = _pivot(df, "Position_Pose", "Mutation", energy_col)
    positions = list(pivot.index)
    wt_map    = (df.drop_duplicates("Position_Pose")
                   .set_index("Position_Pose")["WT"].to_dict()
                 if "WT" in df.columns else {})

    lim  = _sym_lim(pivot.values) if vmin is None else max(abs(vmin), abs(vmax))
    vmin, vmax = -lim, lim

    # ── heatmap ──────────────────────────────────────────────────────────────
    # ── heatmap  ──────────────────────────────────────────────────────
    n_aa  = len(pivot.columns)
    n_pos = len(positions)
    fig_h = max(6, n_aa * 0.40 + 1.5)         # height scales with AA rows
    fig_w = min(24, max(10, n_pos * 0.18 + 3)) # width capped at 24 in
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [fig_w - 0.8, 0.4], "wspace": 0.03},
    )
    ax, cax = axes
    im = ax.imshow(pivot.T.values, aspect="auto",
                   cmap=CMAP_DIV, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(pivot.columns, fontsize=10)
    ax.set_ylabel("Mutant AA", fontsize=12, labelpad=6)
    _xticks(ax, positions, wt_map)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, highlight_positions)
    ax.set_title("Rosetta DMS — ΔΔG per variant", fontsize=13, fontweight="bold", pad=10)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ΔΔG (kcal/mol)", fontsize=10, labelpad=8)
    cax.yaxis.set_tick_params(labelsize=9)
    if highlight_positions:
        ax.legend(handles=[
            mpatches.Patch(facecolor=HL_COLOR, edgecolor=HL_EDGE,
                           alpha=0.4, label=f"Highlighted  {HL_STAR}")
        ], fontsize=9, loc="upper right")
    _save(fig, f"{output_prefix}_rosetta_heatmap.png" if output_prefix else None)
    # ── barplot ───────────────────────────────────────────────────────────────
    site_mean = df.groupby("Position_Pose")[energy_col].mean().reindex(positions)
    fig, ax  = plt.subplots(
        figsize=(min(24, max(10, len(positions) * 0.18 + 3)), 4))
    colors    = ["#e63946" if v > 0 else "#457b9d" for v in site_mean]
    ax.bar(range(len(positions)), site_mean.values, color=colors,
           edgecolor="none", width=0.8)
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_xlim(-0.5, len(positions) - 0.5)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)

    hl_patch = _highlight_bars(ax, positions, site_mean.values,
                                highlight_positions, label=f"Highlighted  {HL_STAR}")
    ax.set_ylabel("Mean ΔΔG (kcal/mol)", fontsize=10)
    ax.set_title("Rosetta DMS — Mean ΔΔG per position", fontsize=12, fontweight="bold")
    legend_handles = [
        Patch(facecolor="#e63946", label="Destabilising (> 0)"),
        Patch(facecolor="#457b9d", label="Stabilising (≤ 0)"),
    ]
    if hl_patch:
        legend_handles.append(hl_patch)
    ax.legend(handles=legend_handles, fontsize=9)
    _save(fig, f"{output_prefix}_rosetta_barplot.png" if output_prefix else None)


# ============================================================================
# 2.  ESM-2 evolutionary scores  →  2 figures
# ============================================================================

def plot_evo_scores(df, output_prefix=None, vmin=None, vmax=None,
                    highlight_positions=None):
    """
    Parameters
    ----------
    highlight_positions : list of int, optional
        1-based positions to mark (same numbering as Position_1based column).
    """
    if "delta_psi_evo" not in df.columns:
        raise ValueError("Column 'delta_psi_evo' not found.")

    pivot     = _pivot(df, "Position_1based", "Mutation", "delta_psi_evo")
    positions = list(pivot.index)
    wt_map    = (df.drop_duplicates("Position_1based")
                   .set_index("Position_1based")["WT"].to_dict()
                 if "WT" in df.columns else {})

    lim  = _sym_lim(pivot.values) if vmin is None else max(abs(vmin), abs(vmax))
    vmin, vmax = -lim, lim

    # ── heatmap ───────────────────────────────────────────────────────────────
    n_aa  = len(pivot.columns)   # 20
    n_pos = len(positions)
    # Height: fixed per-AA row height (0.45 in) + margins; width: per-position
    fig_h = max(6, n_aa * 0.40 + 1.5)         # height scales with AA rows
    fig_w = min(24, max(10, n_pos * 0.18 + 3)) # width capped at 24 in

    fig, axes = plt.subplots(
        1, 2,
        figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [fig_w - 0.8, 0.4], "wspace": 0.03},
    )
    ax, cax = axes

    im = ax.imshow(pivot.T.values, aspect="auto",
                   cmap=CMAP_DIV, vmin=vmin, vmax=vmax, interpolation="nearest")

    # Y-axis: every AA label, adequate font size
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(pivot.columns, fontsize=10)
    ax.set_ylabel("Mutant AA", fontsize=12, labelpad=6)

    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, highlight_positions)
    ax.set_title("ESM-2 — ΔΨᵉᵛᵒ per variant", fontsize=13, fontweight="bold", pad=10)

    # Colorbar in dedicated axis — never steals space from heatmap
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ΔΨᵉᵛᵒ (higher = more deleterious)", fontsize=10, labelpad=8)
    cax.yaxis.set_tick_params(labelsize=9)

    if highlight_positions:
        ax.legend(handles=[
            mpatches.Patch(facecolor=HL_COLOR, edgecolor=HL_EDGE,
                           alpha=0.4, label=f"Highlighted  {HL_STAR}")
        ], fontsize=9, loc="upper right")
    _save(fig, f"{output_prefix}_evo_heatmap.png" if output_prefix else None)

    # ── per-AA mean ───────────────────────────────────────────────────────────
    aa_mean = (df.groupby("Mutation")["delta_psi_evo"]
                 .mean()
                 .reindex([a for a in AA_ORDER if a in df["Mutation"].unique()]))

    fig, ax = plt.subplots(figsize=(5, 7))
    colors  = ["#e63946" if v > 0 else "#457b9d" for v in aa_mean.values]
    ax.barh(range(len(aa_mean)), aa_mean.values, color=colors,
            edgecolor="none", height=0.75)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(range(len(aa_mean)))
    ax.set_yticklabels(aa_mean.index, fontsize=10)
    ax.set_ylim(-0.5, len(aa_mean) - 0.5)
    ax.set_xlabel("Mean ΔΨᵉᵛᵒ", fontsize=11)
    ax.set_title("ESM-2 — Mean ΔΨᵉᵛᵒ\nper mutant AA", fontsize=12, fontweight="bold")
    ax.legend(handles=[
        Patch(facecolor="#e63946", label="Deleterious (> 0)"),
        Patch(facecolor="#457b9d", label="Tolerated (≤ 0)"),
    ], fontsize=9)
    _save(fig, f"{output_prefix}_evo_aa_mean.png" if output_prefix else None)


# ============================================================================
# 3.  Dark energy  →  4 figures
# ============================================================================

def plot_dark_energy(df, output_prefix=None, threshold=None,
                     highlight_positions=None):
    """
    Parameters
    ----------
    highlight_positions : list of int, optional
        1-based positions to mark (e.g. catalytic residues).
        Shown in all four panels.
    """
    required = {"ddG_total_energy", "delta_psi_evo_scaled",
                "dark_energy", "Position_1based", "WT", "Mutation"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {missing}")

    if threshold is None:
        threshold = float(np.percentile(df["dark_energy"], 75))

    tsel    = df["kB_Tsel"].iloc[0] if "kB_Tsel" in df.columns else None
    de_vals = df["dark_energy"].values
    vabs    = _sym_lim(de_vals)
    norm    = mcolors.TwoSlopeNorm(vmin=-vabs, vcenter=0, vmax=vabs)

    pivot     = _pivot(df, "Position_1based", "Mutation", "dark_energy")
    positions = list(pivot.index)
    wt_map    = df.drop_duplicates("Position_1based").set_index("Position_1based")["WT"].to_dict()
    hl        = set(highlight_positions or [])

    # ── landscape scatter ─────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 6))

    # non-highlighted points
    mask_hl = df["Position_1based"].isin(hl)
    ax.scatter(df.loc[~mask_hl, "ddG_total_energy"],
               df.loc[~mask_hl, "delta_psi_evo_scaled"],
               c=de_vals[~mask_hl.values], cmap=CMAP_DIV, norm=norm,
               s=14, alpha=0.5, linewidths=0, label="_nolegend_")

    # highlighted points on top
    if hl:
        sc_hl = ax.scatter(df.loc[mask_hl, "ddG_total_energy"],
                           df.loc[mask_hl, "delta_psi_evo_scaled"],
                           c=de_vals[mask_hl.values], cmap=CMAP_DIV, norm=norm,
                           s=60, alpha=1.0, linewidths=1.2,
                           edgecolors=HL_EDGE, marker="*",
                           label=f"Highlighted  {HL_STAR}")
        ax.legend(fontsize=9)

    lim = max(_sym_lim(df["ddG_total_energy"].values, 98),
              _sym_lim(df["delta_psi_evo_scaled"].values, 98)) * 1.05
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=1,
            label="T_sel line" + (f"  kBT_sel={tsel:.2f}" if tsel else ""))
    ax.axhline(0, color="grey", lw=0.5, ls=":")
    ax.axvline(0, color="grey", lw=0.5, ls=":")
    ax.set_xlabel("ΔEfold  Rosetta ΔΔG (kcal/mol)", fontsize=11)
    ax.set_ylabel("kBT_sel · ΔΨᵉᵛᵒ  (kcal/mol)", fontsize=11)
    ax.set_title("Energy Landscape  (colour = ΔEᵈᵃʳᵏ)", fontsize=12, fontweight="bold")
    sc_all = ax.scatter([], [], c=[], cmap=CMAP_DIV, norm=norm)
    fig.colorbar(sc_all, ax=ax, shrink=0.8).set_label("ΔEᵈᵃʳᵏ (kcal/mol)", fontsize=10)
    _save(fig, f"{output_prefix}_de_landscape.png" if output_prefix else None)

    # ── heatmap ───────────────────────────────────────────────────────────────
    # ── heatmap  ──────────────────────────────────────────────────────
    n_aa  = len(pivot.columns)
    n_pos = len(positions)
    fig_h = max(6, n_aa * 0.40 + 1.5)         # height scales with AA rows
    fig_w = min(24, max(10, n_pos * 0.18 + 3)) # width capped at 24 in
    fig, axes = plt.subplots(
        1, 2, figsize=(fig_w, fig_h),
        gridspec_kw={"width_ratios": [fig_w - 0.8, 0.4], "wspace": 0.03},
    )
    ax, cax = axes
    im = ax.imshow(pivot.T.values, aspect="auto",
                   cmap=CMAP_DIV, norm=norm, interpolation="nearest")
    ax.set_yticks(range(n_aa))
    ax.set_yticklabels(pivot.columns, fontsize=10)
    ax.set_ylabel("Mutant AA", fontsize=12, labelpad=6)
    _xticks(ax, positions, wt_map, highlight=highlight_positions)
    _wt_dots(ax, pivot, wt_map)
    _highlight_heatmap_cols(ax, pivot, hl)
    ax.set_title("Dark Energy — ΔEᵈᵃʳᵏ per variant", fontsize=13, fontweight="bold", pad=10)
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("ΔEᵈᵃʳᵏ (kcal/mol)", fontsize=10, labelpad=8)
    cax.yaxis.set_tick_params(labelsize=9)
    if hl:
        ax.legend(handles=[
            mpatches.Patch(facecolor=HL_COLOR, edgecolor=HL_EDGE,
                           alpha=0.4, label=f"Highlighted  {HL_STAR}")
        ], fontsize=9, loc="upper right")
    _save(fig, f"{output_prefix}_de_heatmap.png" if output_prefix else None)
    # ── site-average barplot ──────────────────────────────────────────────────
    site_avg = (df.groupby(["Position_1based", "WT"])["dark_energy"]
                  .mean().reset_index().sort_values("Position_1based"))
    sa_pos   = list(site_avg["Position_1based"])
    sa_wt    = dict(zip(site_avg["Position_1based"], site_avg["WT"]))
    sa_vals  = site_avg["dark_energy"].values

    fig, ax  = plt.subplots(
        figsize=(min(24, max(10, len(sa_pos) * 0.18 + 3)), 4))
    colors   = ["#e63946" if v >= threshold else "#457b9d" for v in sa_vals]
    ax.bar(range(len(sa_pos)), sa_vals, color=colors, edgecolor="none", width=0.85)
    ax.axhline(threshold, color="#f4a261", lw=1.5, ls="--",
               label=f"Threshold = {threshold:.2f} kcal/mol")
    ax.axhline(0, color="black", lw=0.7)
    ax.set_xlim(-0.5, len(sa_pos) - 0.5)
    _xticks(ax, sa_pos, sa_wt, highlight=highlight_positions)

    hl_patch = _highlight_bars(ax, sa_pos, sa_vals,
                                highlight_positions, label=f"Highlighted  {HL_STAR}")

    n_func = (site_avg["dark_energy"] >= threshold).sum()
    pct    = 100 * n_func / len(site_avg)
    ax.text(0.98, 0.97, f"{n_func}/{len(site_avg)} sites ≥ threshold  ({pct:.1f}%)",
            transform=ax.transAxes, ha="right", va="top", fontsize=9,
            color="#e63946",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8))
    ax.set_ylabel("Mean ΔEᵈᵃʳᵏ (kcal/mol)", fontsize=10)
    ax.set_title("Dark Energy — Site-average ΔEᵈᵃʳᵏ", fontsize=12, fontweight="bold")
    legend_handles = [
        Patch(facecolor="#e63946", label=f"Functional (≥ {threshold:.2f})"),
        Patch(facecolor="#457b9d", label="Folding-related"),
    ]
    if hl_patch:
        legend_handles.append(hl_patch)
    ax.legend(handles=legend_handles, fontsize=9)
    _save(fig, f"{output_prefix}_de_site_avg.png" if output_prefix else None)

    # ── distribution ──────────────────────────────────────────────────────────
    vals = df["dark_energy"].dropna().values
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.hist(vals, bins=60, color="#457b9d", edgecolor="none",
            density=True, alpha=0.55, label="All variants")

    # highlighted variants shown as rug plot
    if hl:
        hl_vals = df.loc[df["Position_1based"].isin(hl), "dark_energy"].dropna().values
        ax.plot(hl_vals, np.zeros_like(hl_vals) - 0.005 * ax.get_ylim()[1],
                "|", color=HL_EDGE, markersize=12, markeredgewidth=1.5,
                label=f"Highlighted positions  {HL_STAR}")

    kde = gaussian_kde(vals, bw_method="scott")
    x   = np.linspace(vals.min(), vals.max(), 500)
    ax.plot(x, kde(x), color="#1d3557", lw=2, label="KDE")
    ax.axvline(threshold, color="#e63946", lw=1.8, ls="--",
               label=f"Threshold = {threshold:.2f}")
    ax.axvline(0, color="black", lw=0.8, ls=":")
    pct_above = 100 * (vals >= threshold).mean()
    ax.text(threshold + 0.02 * (vals.max() - vals.min()),
            ax.get_ylim()[1] * 0.88,
            f"{pct_above:.1f}%\nabove", color="#e63946", fontsize=10, va="top")
    ax.set_xlabel("ΔEᵈᵃʳᵏ (kcal/mol)", fontsize=11)
    ax.set_ylabel("Density", fontsize=11)
    ax.set_title("Distribution of ΔEᵈᵃʳᵏ", fontsize=12, fontweight="bold")
    ax.legend(fontsize=9)
    _save(fig, f"{output_prefix}_de_distribution.png" if output_prefix else None)


# ============================================================================
# Convenience wrapper
# ============================================================================

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


# ============================================================================
# CLI
# ============================================================================
if __name__ == "__main__":
    import argparse
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    p = argparse.ArgumentParser(description="DMS visualisation plots from CSV files.")
    p.add_argument("--rosetta-csv",   default=None)
    p.add_argument("--esm-csv",       default=None)
    p.add_argument("--dark-csv",      default=None)
    p.add_argument("--output-prefix", default=None,
                   help="Path prefix for PNGs. Omit to show interactively.")
    p.add_argument("--threshold", type=float, default=None)
    p.add_argument("--highlight-positions", nargs="+", type=int, default=None,
                   metavar="N",
                   help="1-based positions to highlight (e.g. catalytic residues). "
                        "E.g.: --highlight-positions 70 73 130 166 234")
    args = p.parse_args()
    plot_all(
        rosetta_csv=args.rosetta_csv,
        esm_csv=args.esm_csv,
        dark_csv=args.dark_csv,
        output_prefix=args.output_prefix,
        threshold=args.threshold,
        highlight_positions=args.highlight_positions,
    )
