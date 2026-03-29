#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
bfactor_dark_energy.py
──────────────────────
Replaces the B-factor column of a PDB file with the site-average
ΔEᵈᵃʳᵏ values, allowing direct visualisation in PyMOL / ChimeraX.

Residues not present in the dark energy CSV receive a B-factor of 0.0.

Usage:
    python bfactor_dark_energy.py \\
        --pdb       protein.pdb \\
        --dark-csv  dark_energy_report.csv \\
        --output    protein_dark_energy.pdb

    # Use ΔΨᵉᵛᵒ instead of dark energy
    python bfactor_dark_energy.py \\
        --pdb      protein.pdb \\
        --dark-csv evo_scores.csv \\
        --value-col delta_psi_evo \\
        --output   protein_evo.pdb

    # Use Rosetta ddG
    python bfactor_dark_energy.py \\
        --pdb      protein.pdb \\
        --dark-csv DMS_report.csv \\
        --value-col ddG_total_energy \\
        --pos-col  Position_Pose \\
        --output   protein_ddg.pdb

Visualisation in PyMOL:
    PyMOL> load protein_dark_energy.pdb
    PyMOL> spectrum b, blue_white_red, minimum=-2, maximum=2
    PyMOL> show surface

Visualisation in ChimeraX:
    ChimeraX> open protein_dark_energy.pdb
    ChimeraX> color bfactor palette blue:white:red range -2,2
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core: build residue → value map from CSV
# ---------------------------------------------------------------------------
def build_value_map(
    csv_path:  str,
    value_col: str = "dark_energy",
    pos_col:   str = "Position_1based",
    agg:       str = "mean",
) -> dict:
    """
    Reads the dark energy (or any DMS) CSV and returns a dict mapping
    residue position (int) → aggregated value (float).

    Parameters
    ----------
    csv_path  : path to CSV (dark_energy_report.csv, evo_scores.csv, etc.)
    value_col : column to use as B-factor value
    pos_col   : column with residue position (1-based integer)
    agg       : aggregation over mutations per site ('mean', 'max', 'min')
    """
    df = pd.read_csv(csv_path)

    if pos_col not in df.columns:
        raise ValueError(
            f"Position column '{pos_col}' not found. "
            f"Available: {list(df.columns)}"
        )
    if value_col not in df.columns:
        raise ValueError(
            f"Value column '{value_col}' not found. "
            f"Available: {list(df.columns)}"
        )

    site_vals = df.groupby(pos_col)[value_col].agg(agg)
    value_map = site_vals.to_dict()

    log.info(
        "Loaded %d residue values from '%s'  (col=%s, agg=%s)  "
        "range: [%.3f, %.3f]",
        len(value_map), csv_path, value_col, agg,
        min(value_map.values()), max(value_map.values()),
    )
    return value_map


# ---------------------------------------------------------------------------
# Core: write PDB with replaced B-factors
# ---------------------------------------------------------------------------
def write_bfactor_pdb(
    pdb_in:    str,
    pdb_out:   str,
    value_map: dict,
    default:   float = 0.0,
) -> None:
    """
    Reads a PDB line by line and replaces the B-factor field (cols 60-65)
    for every ATOM / HETATM record whose residue sequence number matches
    a key in value_map.

    PDB format (fixed-width):
        cols  1- 6  record type
        cols  7-11  atom serial
        cols 13-16  atom name
        col  17     alt loc
        cols 18-20  residue name
        col  22     chain ID
        cols 23-26  residue seq number  ← matched against value_map
        col  27     insertion code
        cols 31-38  X
        cols 39-46  Y
        cols 47-54  Z
        cols 55-60  occupancy
        cols 61-66  B-factor           ← replaced here
    """
    n_replaced = 0
    n_missing  = 0
    seen_missing = set()

    in_path  = Path(pdb_in)
    out_path = Path(pdb_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with open(in_path) as fin, open(out_path, "w") as fout:
        for line in fin:
            record = line[:6].strip()

            if record in ("ATOM", "HETATM") and len(line) >= 60:
                try:
                    resseq = int(line[22:26].strip())
                except ValueError:
                    fout.write(line)
                    continue

                bval = value_map.get(resseq)
                if bval is None:
                    bval = default
                    if resseq not in seen_missing:
                        seen_missing.add(resseq)
                        n_missing += 1
                else:
                    n_replaced += 1

                # Rebuild line: keep everything except B-factor field (cols 60-65)
                bfactor_str = f"{bval:6.2f}"
                # PDB cols 61-66 = indices 60-65
                new_line = line[:60] + bfactor_str + line[66:]
                fout.write(new_line)
            else:
                fout.write(line)

    log.info(
        "Written → %s  |  %d atoms updated  |  %d residues without data (set to %.1f)",
        out_path, n_replaced, n_missing, default,
    )


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Replace PDB B-factors with DMS/dark energy values.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--pdb", required=True,
                        help="Input PDB file.")
    parser.add_argument("--dark-csv", required=True,
                        help="CSV with per-residue values "
                             "(dark_energy_report.csv, evo_scores.csv, etc.)")
    parser.add_argument("--output", required=True,
                        help="Output PDB file with replaced B-factors.")
    parser.add_argument("--value-col", default="dark_energy",
                        help="Column to use as B-factor. "
                             "Default: 'dark_energy'. "
                             "Other options: 'delta_psi_evo', 'ddG_total_energy'.")
    parser.add_argument("--pos-col", default="Position_1based",
                        help="Column with residue position (1-based integer). "
                             "Default: 'Position_1based'. "
                             "For Rosetta CSV use 'Position_Pose'.")
    parser.add_argument("--agg", default="mean",
                        choices=["mean", "max", "min", "median"],
                        help="How to aggregate multiple mutations per site. "
                             "Default: mean.")
    parser.add_argument("--default", type=float, default=0.0,
                        help="B-factor for residues absent in CSV. Default: 0.0.")
    return parser.parse_args()


def main():
    args = _parse_args()

    if not Path(args.pdb).is_file():
        log.error("PDB not found: %s", args.pdb)
        sys.exit(1)
    if not Path(args.dark_csv).is_file():
        log.error("CSV not found: %s", args.dark_csv)
        sys.exit(1)

    value_map = build_value_map(
        csv_path=args.dark_csv,
        value_col=args.value_col,
        pos_col=args.pos_col,
        agg=args.agg,
    )

    write_bfactor_pdb(
        pdb_in=args.pdb,
        pdb_out=args.output,
        value_map=value_map,
        default=args.default,
    )

    # Print ready-to-use visualisation commands
    lo = min(value_map.values())
    hi = max(value_map.values())
    mid = (lo + hi) / 2

    print("\n── PyMOL ──────────────────────────────────────────────────────")
    print(f"load {args.output}")
    print(f"spectrum b, blue_white_red, minimum={lo:.2f}, maximum={hi:.2f}")
    print("show surface")
    print("ray 1200, 900")

    print("\n── ChimeraX ───────────────────────────────────────────────────")
    print(f"open {args.output}")
    print(f"color bfactor palette blue:white:red range {lo:.2f},{hi:.2f}")
    print("surface")


if __name__ == "__main__":
    main()
