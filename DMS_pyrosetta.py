#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DMS in silico - Deep Mutational Scanning via PyRosetta
Calcula ddG = E(mutante) - E(WT) por resíduo para cada posição especificada.

Usage:
    python dms_insilico.py --pdb input.pdb --positions 10 11 12 --ncpu 4
    python dms_insilico.py --pdb input.pdb --positions 10 11 12 --ncpu 4 --save-structures
    python dms_insilico.py --pdb input.pdb --positions-file positions.txt --ncpu 8

    # Com FastRelax após cada mutação (default: 1 repeat)
    python dms_insilico.py --pdb input.pdb --positions 10 11 --ncpu 4 --fast-relax-repeats 1

    # Com mais repeats para maior relaxamento
    python dms_insilico.py --pdb input.pdb --positions 10 11 --ncpu 4 --fast-relax-repeats 3
"""

import argparse
import os
import sys
import logging
import multiprocessing
from pathlib import Path

import pandas as pd
import pyrosetta
from pyrosetta import pose_from_pdb
from rosetta.core.scoring.methods import EnergyMethodOptions
import rosetta.core.scoring

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
# Amino acid constants
# ---------------------------------------------------------------------------
AA_20 = ['G', 'A', 'L', 'M', 'F', 'W', 'K', 'Q', 'E', 'S',
         'P', 'V', 'I', 'C', 'Y', 'H', 'R', 'N', 'D', 'T']

AA_3TO1 = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D",
    "CYS": "C", "GLN": "Q", "GLU": "E", "GLY": "G",
    "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S",
    "THR": "T", "TRP": "W", "TYR": "Y", "VAL": "V",
}

# ---------------------------------------------------------------------------
# Score function factory  (one per process)
# ---------------------------------------------------------------------------
def _make_scorefxn():
    """Creates ref2015_cart score function with hbond pair decomposition."""
    sfxn = pyrosetta.create_score_function("ref2015_cart.wts")
    emopts = EnergyMethodOptions(sfxn.energy_method_options())
    emopts.hbond_options().decompose_bb_hb_into_pair_energies(True)
    sfxn.set_energy_method_options(emopts)
    return sfxn


# ---------------------------------------------------------------------------
# Pose / PDB utilities
# ---------------------------------------------------------------------------
def pdb_pose_dictionary(pose) -> pd.DataFrame:
    """Maps Pose numbering ↔ PDB numbering for all residues."""
    records = []
    for i in range(1, pose.total_residue() + 1):
        chain = pose.pdb_info().chain(i)
        pdb_num = pose.pdb_info().number(i)
        records.append({"Chain": chain, "IndexPDB": pdb_num, "IndexPose": i})
    return pd.DataFrame(records)


# ---------------------------------------------------------------------------
# Energy contribution
# ---------------------------------------------------------------------------
def energy_contribution(pose, scorefxn) -> pd.DataFrame:
    """
    Returns a DataFrame with per-residue weighted energy terms.

    Columns: Residue_Index_Pose, Residue_Index_PDB, Residue_Name,
             Residue_Name1, Chain, <energy_term_1>, ..., total_energy
    """
    scorefxn.score(pose)
    weights = scorefxn.weights()

    # Collect non-zero weighted score types once
    weighted_types = [
        st for st in rosetta.core.scoring.ScoreType.__members__.values()
        if weights[st] != 0
    ]

    idx_map = pdb_pose_dictionary(pose)
    data = []

    for i in range(1, pose.total_residue() + 1):
        res_energies = pose.energies().residue_total_energies(i)
        row = {
            "Residue_Index_Pose": i,
            "Residue_Index_PDB":  idx_map.loc[i - 1, "IndexPDB"],
            "Chain":              idx_map.loc[i - 1, "Chain"],
            "Residue_Name":       pose.residue(i).name(),
        }
        total = 0.0
        for st in weighted_types:
            val = res_energies[st] * weights[st]
            row[st.name] = val
            total += val
        row["total_energy"] = total
        data.append(row)

    df = pd.DataFrame(data)
    df.insert(3, "Residue_Name1",
              df["Residue_Name"].str[:3].map(AA_3TO1))

    # Reorder: metadata first, then energy terms
    meta_cols = ["Residue_Index_Pose", "Residue_Index_PDB",
                 "Residue_Name", "Residue_Name1", "Chain"]
    energy_cols = [c for c in df.columns if c not in meta_cols]
    return df[meta_cols + energy_cols]


# ---------------------------------------------------------------------------
# Mutation & repack
# ---------------------------------------------------------------------------
def mutate_repack(pose, position: int, amino: str, scorefxn):
    """
    Clones pose, mutates residue at *position* (Pose numbering) to *amino*,
    repacks neighborhood, and returns the mutated clone.
    """
    new_pose = pose.clone()

    mut_sel = pyrosetta.rosetta.core.select.residue_selector.ResidueIndexSelector()
    mut_sel.set_index(position)

    nbr_sel = pyrosetta.rosetta.core.select.residue_selector.NeighborhoodResidueSelector()
    nbr_sel.set_focus_selector(mut_sel)
    nbr_sel.set_include_focus_in_subset(True)

    not_design = pyrosetta.rosetta.core.select.residue_selector.NotResidueSelector(mut_sel)

    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.IncludeCurrent())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.NoRepackDisulfides())

    # Prevent repacking outside neighborhood
    prevent_rlt = pyrosetta.rosetta.core.pack.task.operation.PreventRepackingRLT()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        prevent_rlt, nbr_sel, True))

    # Repack only (no design) for non-mutated residues in neighborhood
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        pyrosetta.rosetta.core.pack.task.operation.RestrictToRepackingRLT(), not_design))

    # Design: restrict mutation site to target amino acid
    aa_rlt = pyrosetta.rosetta.core.pack.task.operation.RestrictAbsentCanonicalAASRLT()
    aa_rlt.aas_to_keep(amino)
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.OperateOnResidueSubset(
        aa_rlt, mut_sel))

    packer = pyrosetta.rosetta.protocols.minimization_packing.PackRotamersMover(scorefxn)
    packer.task_factory(tf)
    packer.apply(new_pose)

    return new_pose


# ---------------------------------------------------------------------------
# FastRelax (optional post-mutation relaxation)
# ---------------------------------------------------------------------------
def fast_relax(pose, scorefxn, repeats: int = 1) -> None:
    """
    Applies a cartesian FastRelax protocol to *pose* in-place.

    Parameters
    ----------
    pose : pyrosetta Pose
        Pose to relax (modified in-place).
    scorefxn : ScoreFunction
        ref2015_cart-compatible score function.
    repeats : int
        Number of FastRelax rounds (standard_repeats). Default: 1.
        Higher values improve geometry but increase runtime significantly.
    """
    tf = pyrosetta.rosetta.core.pack.task.TaskFactory()
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.InitializeFromCommandline())
    tf.push_back(pyrosetta.rosetta.core.pack.task.operation.RestrictToRepacking())

    mmf = pyrosetta.rosetta.core.select.movemap.MoveMapFactory()
    mmf.all_bb(setting=True)
    mmf.all_bondangles(setting=True)
    mmf.all_bondlengths(setting=True)
    mmf.all_chi(setting=True)
    mmf.all_jumps(setting=True)
    mmf.set_cartesian(setting=True)

    fr = pyrosetta.rosetta.protocols.relax.FastRelax(
        scorefxn_in=scorefxn, standard_repeats=repeats
    )
    fr.cartesian(True)
    fr.set_task_factory(tf)
    fr.set_movemap_factory(mmf)
    fr.min_type("lbfgs_armijo_nonmonotone")
    fr.apply(pose)


# ---------------------------------------------------------------------------
# Per-position DMS worker  (runs inside a subprocess)
# ---------------------------------------------------------------------------
def _dms_worker(args: tuple) -> str:
    """
    Worker function executed by each subprocess.

    Parameters
    ----------
    args : tuple
        (pdb_path, position, save_structures, output_dir, fast_relax_repeats)

    Returns
    -------
    str : path to the CSV written by this worker.
    """
    pdb_path, position, save_structures, output_dir, fast_relax_repeats = args

    # PyRosetta must be initialised inside each subprocess
    pyrosetta.init(
        options="-mute all",
        set_logging_handler="logging",
        extra_options="",
    )

    scorefxn = _make_scorefxn()
    pose_wt  = pose_from_pdb(pdb_path)
    scorefxn.score(pose_wt)

    wt_energy_df = energy_contribution(pose_wt, scorefxn)
    wt_residue   = pose_wt.residue(position).name()[:3]
    wt_aa1       = AA_3TO1.get(wt_residue, "X")

    pdb_dir = Path(output_dir) / "structures"
    csv_dir = Path(output_dir) / "csv"
    if save_structures:
        pdb_dir.mkdir(parents=True, exist_ok=True)
    csv_dir.mkdir(parents=True, exist_ok=True)

    rows = []

    for aa in AA_20:
        mut_pose = mutate_repack(pose_wt, position, aa, scorefxn)

        if fast_relax_repeats > 0:
            log.debug("  FastRelax %s (repeats=%d)", aa, fast_relax_repeats)
            fast_relax(mut_pose, scorefxn, repeats=fast_relax_repeats)

        mut_energy = energy_contribution(mut_pose, scorefxn)

        # ddG per residue per energy term  =  E_mut - E_wt
        energy_cols = [c for c in wt_energy_df.columns
                       if c not in ("Residue_Index_Pose", "Residue_Index_PDB",
                                    "Residue_Name", "Residue_Name1", "Chain")]

        ddg_df = mut_energy[energy_cols].subtract(wt_energy_df[energy_cols])

        # One summary row per mutation: total ddG (sum over all residues)
        summary = {
            "Position_Pose":   position,
            "Position_PDB":    wt_energy_df.loc[position - 1, "Residue_Index_PDB"],
            "Chain":           wt_energy_df.loc[position - 1, "Chain"],
            "WT":              wt_aa1,
            "Mutation":        aa,
            "Label":           f"{wt_aa1}{wt_energy_df.loc[position - 1, 'Residue_Index_PDB']}{aa}",
        }
        for col in energy_cols:
            summary[f"ddG_{col}"] = ddg_df[col].sum()

        rows.append(summary)

        if save_structures:
            out_pdb = pdb_dir / f"{position}_{wt_aa1}{wt_energy_df.loc[position-1,'Residue_Index_PDB']}{aa}.pdb"
            mut_pose.dump_pdb(str(out_pdb))
            log.info("  saved %s", out_pdb.name)

    df_out = pd.DataFrame(rows)

    csv_path = csv_dir / f"DMS_pos{position}.csv"
    df_out.to_csv(csv_path, index=False)
    log.info("Position %d done → %s", position, csv_path)
    return str(csv_path)


# ---------------------------------------------------------------------------
# Main runner
# ---------------------------------------------------------------------------
def run_dms(pdb: str,
            positions: list[int],
            n_cpu: int,
            save_structures: bool,
            output_dir: str,
            fast_relax_repeats: int = 0) -> pd.DataFrame:
    """
    Runs in-silico DMS for *positions* in parallel using multiprocessing.Pool.

    Parameters
    ----------
    fast_relax_repeats : int
        Number of FastRelax rounds after each mutation. 0 = disabled (default).

    Returns a consolidated DataFrame with ddG for every mutation × position.
    """
    log.info(
        "Starting DMS | PDB: %s | Positions: %s | CPUs: %d | "
        "Save structures: %s | FastRelax repeats: %d",
        pdb, positions, n_cpu, save_structures, fast_relax_repeats,
    )

    worker_args = [
        (pdb, pos, save_structures, output_dir, fast_relax_repeats)
        for pos in positions
    ]

    with multiprocessing.Pool(processes=n_cpu) as pool:
        csv_paths = pool.map(_dms_worker, worker_args)

    # Consolidate all per-position CSVs into one report
    dfs = [pd.read_csv(p) for p in csv_paths]
    df_all = pd.concat(dfs, ignore_index=True)

    report_path = Path(output_dir) / "DMS_report.csv"
    df_all.to_csv(report_path, index=False)
    log.info("Consolidated report saved → %s", report_path)
    return df_all


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args():
    parser = argparse.ArgumentParser(
        description="In-silico Deep Mutational Scanning via PyRosetta",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    # Input
    parser.add_argument("--pdb", required=True,
                        help="Path to input PDB file.")

    pos_group = parser.add_mutually_exclusive_group(required=False)
    pos_group.add_argument(
        "--positions", nargs="+", type=int,
        metavar="N",
        help=(
            "Pose residue indices to scan (space-separated). "
            "E.g.: --positions 10 11 12. "
            "Default: all residues in the structure."
        ),
    )
    pos_group.add_argument(
        "--positions-file",
        metavar="FILE",
        help="Text file with one Pose residue index per line.",
    )

    # Compute
    parser.add_argument(
        "--ncpu", type=int, default=max(1, multiprocessing.cpu_count() - 1),
        help="Number of parallel worker processes (default: all CPUs - 1).",
    )

    # Output
    parser.add_argument(
        "--output-dir", default="./DMS_output",
        help="Directory for CSV reports and (optionally) PDB structures.",
    )
    parser.add_argument(
        "--save-structures", action="store_true",
        help="Save PDB files for every mutant structure.",
    )
    parser.add_argument(
        "--fast-relax-repeats", type=int, default=0, metavar="N",
        help=(
            "Number of FastRelax rounds applied after each mutation. "
            "0 = disabled (default). "
            "1 is usually sufficient; higher values improve geometry "
            "but increase runtime linearly."
        ),
    )
    parser.add_argument(
        "--plot", action="store_true",
        help="Generate visualisation plots after the DMS run.",
    )
    parser.add_argument(
        "--plot-prefix", default=None, metavar="PATH",
        help=(
            "File path prefix for saved plots (e.g. ./output/myprotein). "
            "Omit to show interactively when --plot is set."
        ),
    )

    return parser.parse_args()


def _all_residues_from_pdb(pdb_path: str) -> list[int]:
    """Returns all Pose residue indices (1-based) for a PDB file."""
    pyrosetta.init(options="-mute all")
    pose = pose_from_pdb(pdb_path)
    return list(range(1, pose.total_residue() + 1))


def main():
    args = parse_args()

    if not os.path.isfile(args.pdb):
        log.error("PDB file not found: %s", args.pdb)
        sys.exit(1)

    # Resolve positions — default: all residues in the structure
    if args.positions:
        positions = args.positions
    elif args.positions_file:
        with open(args.positions_file) as fh:
            positions = [int(line.strip()) for line in fh if line.strip()]
    else:
        log.info("No positions specified — scanning all residues in %s", args.pdb)
        positions = _all_residues_from_pdb(args.pdb)
        log.info("Total residues to scan: %d", len(positions))

    if not positions:
        log.error("Position list is empty. Aborting.")
        sys.exit(1)

    if not os.path.isfile(args.pdb):
        log.error("PDB file not found: %s", args.pdb)
        sys.exit(1)

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    df = run_dms(
        pdb=args.pdb,
        positions=positions,
        n_cpu=args.ncpu,
        save_structures=args.save_structures,
        output_dir=args.output_dir,
        fast_relax_repeats=args.fast_relax_repeats,
    )

    log.info("Done. %d mutation rows in report.", len(df))

    if args.plot:
        from dms_plots import plot_rosetta_dms
        log.info("Generating Rosetta DMS plot…")
        plot_rosetta_dms(df, output_prefix=args.plot_prefix)


if __name__ == "__main__":
    # Required for multiprocessing on macOS / Windows
    multiprocessing.set_start_method("spawn", force=True)
    main()