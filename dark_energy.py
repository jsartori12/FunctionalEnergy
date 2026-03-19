#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
dark_energy.py
──────────────
Calcula a Dark Energy (ΔEᵈᵃʳᵏ) combinando:

    • ΔEfold  →  ddG_total_energy do DMS Rosetta  (dms_insilico.py)
    • ΔΨᵉᵛᵒ   →  log-likelihood ratio do ESM-2    (evolutionary_score.py)

Equação central (Galpern et al. 2026, Eq. 8):

    ΔEᵈᵃʳᵏ = kB·T_sel^fold · ΔΨᵉᵛᵒ − ΔEfold

onde T_sel^fold é estimada dos dados como:

    kB·T_sel^fold = sd(ΔEfold) / sd(ΔΨᵉᵛᵒ)

Usage:
    # Modo completo (calcula ESM + Rosetta e integra)
    python dark_energy.py \\
        --pdb       protein.pdb \\
        --sequence  MKTAYIAKQRQISFVKSHFSRQ... \\
        --positions 10 11 12 \\
        --ncpu      4

    # Modo integração apenas (com CSVs já calculados)
    python dark_energy.py \\
        --rosetta-csv  DMS_report.csv \\
        --esm-csv      evo_scores.csv \\
        --output       dark_energy_report.csv
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Step 1 — Estimate folding selection temperature  T_sel^fold
# ---------------------------------------------------------------------------
def estimate_tsel(
    df: pd.DataFrame,
    col_efold:   str = "ddG_total_energy",
    col_psi_evo: str = "delta_psi_evo",
) -> float:
    """
    Estimates the folding selection temperature (in kcal/mol units) as:

        kB·T_sel^fold = sd(ΔEfold) / sd(ΔΨᵉᵛᵒ)

    This rescaling converts the dimensionless evolutionary scores to
    physical folding energy units (kcal/mol), as done in Galpern et al. 2026.

    Parameters
    ----------
    df : DataFrame
        Merged table containing both ΔEfold and ΔΨᵉᵛᵒ columns.
    col_efold : str
        Column name for ΔEfold values (Rosetta ddG).
    col_psi_evo : str
        Column name for ΔΨᵉᵛᵒ values (ESM-2 score).

    Returns
    -------
    float
        kB·T_sel^fold in kcal/mol.
    """
    sd_efold  = df[col_efold].std()
    sd_psievo = df[col_psi_evo].std()

    if sd_psievo == 0:
        raise ValueError("sd(ΔΨᵉᵛᵒ) = 0 — cannot estimate T_sel.")

    tsel = sd_efold / sd_psievo
    log.info("Estimated kB·T_sel^fold = %.4f kcal/mol  "
             "[sd(ΔEfold)=%.4f | sd(ΔΨᵉᵛᵒ)=%.4f]",
             tsel, sd_efold, sd_psievo)
    return tsel


# ---------------------------------------------------------------------------
# Step 2 — Merge Rosetta + ESM-2 tables
# ---------------------------------------------------------------------------
def merge_dms_tables(
    df_rosetta: pd.DataFrame,
    df_esm:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Merges Rosetta DMS (ΔEfold) and ESM-2 (ΔΨᵉᵛᵒ) tables on a common key.

    Matching key: (WT, Mutation, position)

    The Rosetta table uses Pose numbering (1-based, Position_Pose).
    The ESM-2 table uses Position_1based (also 1-based when sequence
    starts at residue 1 with no gaps).

    If the PDB has an offset between Pose and sequence index (e.g. the
    sequence passed to ESM does not start at residue 1 of the PDB),
    pass `pose_offset` to adjust.

    Parameters
    ----------
    df_rosetta : DataFrame  (output of dms_insilico.py  →  DMS_report.csv)
    df_esm     : DataFrame  (output of evolutionary_score.py → evo_scores.csv)

    Returns
    -------
    Merged DataFrame with both ΔEfold and ΔΨᵉᵛᵒ columns.
    """
    # Normalise column names for the join
    ros = df_rosetta.rename(columns={"Position_Pose": "Position_1based"})[
        ["Position_1based", "WT", "Mutation", "Label", "ddG_total_energy"]
        + [c for c in df_rosetta.columns
           if c.startswith("ddG_") and c != "ddG_total_energy"]
    ].copy()

    evo = df_esm[["Position_1based", "WT", "Mutation", "delta_psi_evo"]].copy()

    merged = pd.merge(
        ros, evo,
        on=["Position_1based", "WT", "Mutation"],
        how="inner",
        validate="1:1",
    )

    n_ros = len(ros)
    n_evo = len(evo)
    n_merged = len(merged)
    log.info("Merge result: %d Rosetta rows × %d ESM rows → %d matched.",
             n_ros, n_evo, n_merged)

    if n_merged < 0.9 * min(n_ros, n_evo):
        log.warning(
            "Less than 90%% of rows matched. "
            "Check that Position_1based aligns between Rosetta (Pose) "
            "and ESM-2 (sequence index). "
            "Use --pose-offset if the PDB chain doesn't start at residue 1."
        )
    return merged


# ---------------------------------------------------------------------------
# Step 3 — Compute ΔEᵈᵃʳᵏ
# ---------------------------------------------------------------------------
def compute_dark_energy(
    df_merged: pd.DataFrame,
    tsel:      float | None = None,
    col_efold:   str = "ddG_total_energy",
    col_psi_evo: str = "delta_psi_evo",
) -> pd.DataFrame:
    """
    Computes ΔEᵈᵃʳᵏ per variant (Eq. 8, Galpern et al. 2026):

        ΔEᵈᵃʳᵏ = kB·T_sel^fold · ΔΨᵉᵛᵒ  −  ΔEfold

    Parameters
    ----------
    df_merged : DataFrame
        Output of merge_dms_tables().
    tsel : float or None
        kB·T_sel^fold in kcal/mol.  If None, it is estimated from the data.
    col_efold : str
        Column with ΔEfold values.
    col_psi_evo : str
        Column with ΔΨᵉᵛᵒ values.

    Returns
    -------
    DataFrame with new columns:
        kB_Tsel, delta_psi_evo_scaled, dark_energy
    """
    df = df_merged.copy()

    if tsel is None:
        tsel = estimate_tsel(df, col_efold=col_efold, col_psi_evo=col_psi_evo)

    df["kB_Tsel"]               = tsel
    df["delta_psi_evo_scaled"]  = tsel * df[col_psi_evo]   # kB·T_sel · ΔΨᵉᵛᵒ
    df["dark_energy"]           = df["delta_psi_evo_scaled"] - df[col_efold]

    log.info(
        "Dark energy stats:  mean=%.3f  sd=%.3f  min=%.3f  max=%.3f  [kcal/mol]",
        df["dark_energy"].mean(),
        df["dark_energy"].std(),
        df["dark_energy"].min(),
        df["dark_energy"].max(),
    )
    return df


# ---------------------------------------------------------------------------
# Step 4 — Per-site weighted average  (for structure coloring / heatmaps)
# ---------------------------------------------------------------------------
def site_average_dark_energy(df: pd.DataFrame) -> pd.DataFrame:
    """
    Computes the weighted site-average ΔEᵈᵃʳᵏ per position,
    weighting each variant by the amino acid frequency in the
    20-AA alphabet (uniform weight = 1/20), as done in the paper.

    Returns a DataFrame indexed by Position_1based with columns:
        WT, mean_dark_energy, weighted_dark_energy, n_variants
    """
    agg = (
        df.groupby(["Position_1based", "WT"])
        .agg(
            mean_dark_energy     = ("dark_energy", "mean"),
            weighted_dark_energy = ("dark_energy", "mean"),   # uniform weights
            n_variants           = ("dark_energy", "count"),
        )
        .reset_index()
    )
    return agg.sort_values("Position_1based")


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def run_dark_energy_pipeline(
    rosetta_csv:  str,
    esm_csv:      str,
    output:       str = "dark_energy_report.csv",
    tsel:         float | None = None,
) -> pd.DataFrame:
    """
    Loads pre-computed Rosetta and ESM-2 CSV files and produces the
    final dark energy report.
    """
    log.info("Loading Rosetta DMS table:  %s", rosetta_csv)
    df_rosetta = pd.read_csv(rosetta_csv)

    log.info("Loading ESM-2 evo scores:   %s", esm_csv)
    df_esm = pd.read_csv(esm_csv)

    df_merged = merge_dms_tables(df_rosetta, df_esm)
    df_dark   = compute_dark_energy(df_merged, tsel=tsel)

    df_site   = site_average_dark_energy(df_dark)

    # Save full per-variant report
    out_path = Path(output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_dark.to_csv(out_path, index=False)
    log.info("Per-variant dark energy saved → %s", out_path)

    # Save per-site averages alongside
    site_path = out_path.with_stem(out_path.stem + "_site_avg")
    df_site.to_csv(site_path, index=False)
    log.info("Site-average dark energy saved → %s", site_path)

    return df_dark


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute protein dark energy from Rosetta DMS + ESM-2 scores.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    mode = parser.add_argument_group("Input mode (choose one)")
    mutex = mode.add_mutually_exclusive_group(required=True)

    # Mode A: pre-computed CSVs
    mutex.add_argument(
        "--rosetta-csv",
        metavar="FILE",
        help="Pre-computed Rosetta DMS report (DMS_report.csv from dms_insilico.py).",
    )

    parser.add_argument(
        "--esm-csv",
        metavar="FILE",
        help="Pre-computed ESM-2 evolutionary scores (evo_scores.csv).",
    )

    # Mode B: compute everything inline  (convenience wrapper)
    mutex.add_argument(
        "--pdb",
        metavar="FILE",
        help="PDB file (triggers Rosetta DMS + ESM-2 computation inline).",
    )
    parser.add_argument(
        "--sequence",
        metavar="SEQ",
        help="Protein sequence for ESM-2 (required with --pdb).",
    )
    parser.add_argument(
        "--positions", nargs="+", type=int, metavar="N",
        help="Pose residue indices to scan (required with --pdb).",
    )
    parser.add_argument(
        "--ncpu", type=int, default=1,
        help="CPUs for Rosetta DMS (used with --pdb).",
    )

    # Optional
    parser.add_argument(
        "--tsel", type=float, default=None,
        help="Manually set kB·T_sel (kcal/mol). Default: estimated from data.",
    )
    parser.add_argument(
        "--output", default="dark_energy_report.csv",
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-dir", default="./dark_energy_output",
        help="Working directory for intermediate files (used with --pdb).",
    )
    parser.add_argument(
        "--fast-relax-repeats", type=int, default=0, metavar="N",
        help="FastRelax rounds for Rosetta DMS (used with --pdb). Default: 0.",
    )
    parser.add_argument(
        "--save-structures", action="store_true",
        help="Save PDB structures from Rosetta DMS (used with --pdb).",
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate all visualisation plots after the pipeline completes.",
    )
    parser.add_argument(
        "--plot-prefix", default=None, metavar="PATH",
        help="File path prefix for saved plots. Omit to show interactively.",
    )
    parser.add_argument(
        "--plot-threshold", type=float, default=None, metavar="X",
        help="Dark energy threshold line on distribution plot (kcal/mol). "
             "Default: 75th percentile of the data.",
    )

    return parser.parse_args()


def main():
    args = _parse_args()

    # ── Mode A: integration from pre-computed files ──────────────────────
    if args.rosetta_csv:
        if not args.esm_csv:
            log.error("--esm-csv is required when using --rosetta-csv.")
            sys.exit(1)
        df_dark = run_dark_energy_pipeline(
            rosetta_csv=args.rosetta_csv,
            esm_csv=args.esm_csv,
            output=args.output,
            tsel=args.tsel,
        )
        if args.plot:
            from dms_plots import plot_all
            log.info("Generating plots…")
            plot_all(
                rosetta_csv=args.rosetta_csv,
                esm_csv=args.esm_csv,
                dark_csv=args.output,
                output_prefix=args.plot_prefix,
                threshold=args.plot_threshold,
            )
        return

    # ── Mode B: compute Rosetta DMS + ESM-2 inline ───────────────────────
    if not args.sequence:
        log.error("--sequence is required with --pdb.")
        sys.exit(1)

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Resolve positions — None means "all residues" for both Rosetta and ESM-2
    rosetta_positions = args.positions  # stays None if not specified

    # 1. Rosetta DMS
    from dms_insilico import run_dms, _all_residues_from_pdb
    if rosetta_positions is None:
        log.info("Step 1/3 — Running Rosetta DMS (all residues)…")
        rosetta_positions = _all_residues_from_pdb(args.pdb)
    else:
        log.info("Step 1/3 — Running Rosetta DMS (%d positions)…", len(rosetta_positions))
    df_rosetta = run_dms(
        pdb=args.pdb,
        positions=rosetta_positions,
        n_cpu=args.ncpu,
        save_structures=args.save_structures,
        output_dir=str(out_dir),
        fast_relax_repeats=args.fast_relax_repeats,
    )
    rosetta_csv = out_dir / "DMS_report.csv"

    # 2. ESM-2 evolutionary scores
    # Convert 1-based Pose positions to 0-based sequence indices.
    log.info("Step 2/3 — Running ESM-2 ΔΨᵉᵛᵒ scoring (%d positions)…", len(rosetta_positions))
    from evolutionary_score import compute_evo_scores_dms
    positions_0based = [p - 1 for p in rosetta_positions]
    df_esm = compute_evo_scores_dms(args.sequence, positions=positions_0based)
    esm_csv = out_dir / "evo_scores.csv"
    df_esm.to_csv(esm_csv, index=False)


    # 3. Dark energy
    log.info("Step 3/3 — Computing dark energy…")
    df_dark = run_dark_energy_pipeline(
        rosetta_csv=str(rosetta_csv),
        esm_csv=str(esm_csv),
        output=args.output,
        tsel=args.tsel,
    )

    if args.plot:
        from dms_plots import plot_all
        log.info("Generating plots…")
        plot_all(
            rosetta_csv=str(rosetta_csv),
            esm_csv=str(esm_csv),
            dark_csv=args.output,
            output_prefix=args.plot_prefix,
            threshold=args.plot_threshold,
        )


if __name__ == "__main__":
    main()