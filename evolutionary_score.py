#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
evolutionary_score.py
─────────────────────
Calcula o score evolutivo ΔΨᵉᵛᵒ (log-likelihood ratio) via ESM-2 para
todas as substituições de um único sítio, conforme Galpern et al. 2026.

A convenção de sinal segue o artigo:
    ΔΨᵉᵛᵒ = −[logit(mutante) − logit(WT)]
           = −log[ P(b | σ_i) / P(a | σ_i) ]

Valores maiores → mutação menos provável sob as restrições evolutivas.

Usage standalone:
    python evolutionary_score.py --sequence MRWQEMGYIFYPRKLR --output evo_scores.csv
    python evolutionary_score.py --sequence MRWQEMGYIFYPRKLR --positions 0 1 2 --output evo_scores.csv
"""




from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
import esm
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
# Amino acid constants  (mesmo conjunto usado no DMS Rosetta)
# ---------------------------------------------------------------------------
AA_20 = ['A', 'R', 'N', 'D', 'C', 'Q', 'E', 'G', 'H', 'I',
         'L', 'K', 'M', 'F', 'P', 'S', 'T', 'W', 'Y', 'V']

# ---------------------------------------------------------------------------
# Model loading  (singleton — carregado uma única vez por processo)
# ---------------------------------------------------------------------------
_ESM_MODEL   = None
_ESM_ALPHABET = None
_BATCH_CONVERTER = None

def _load_model():
    """Loads ESM-2 (650M, 33 layers) once and caches globally."""
    global _ESM_MODEL, _ESM_ALPHABET, _BATCH_CONVERTER
    if _ESM_MODEL is None:
        log.info("Loading ESM-2 model (650M)…")
        _ESM_MODEL, _ESM_ALPHABET = esm.pretrained.esm2_t33_650M_UR50D()
        _ESM_MODEL.eval()
        _BATCH_CONVERTER = _ESM_ALPHABET.get_batch_converter()
        log.info("ESM-2 loaded.")
    return _ESM_MODEL, _ESM_ALPHABET, _BATCH_CONVERTER


# ---------------------------------------------------------------------------
# Core: masked marginal log-likelihood ratio  (ΔΨᵉᵛᵒ)
# ---------------------------------------------------------------------------
def compute_delta_psi_evo(
    sequence: str,
    position: int,        # 0-based index into sequence
) -> dict[str, float]:
    """
    Computes ΔΨᵉᵛᵒ for all 20 standard AAs at a single position.

    The wild-type residue at `position` is masked; the model predicts
    the conditional distribution P(aa | σ_i).  ΔΨᵉᵛᵒ is the
    log-likelihood ratio between the mutant and WT residues:

        ΔΨᵉᵛᵒ(a→b) = −[ logit_i(b) − logit_i(a) ]
                    = −log[ P(b|σ_i) / P(a|σ_i) ]

    Parameters
    ----------
    sequence : str
        Full protein sequence (single-letter codes).
    position : int
        0-based position to mask and score.

    Returns
    -------
    dict  {aa: delta_psi}
        ΔΨᵉᵛᵒ for every AA in AA_20.
        WT residue will have ΔΨᵉᵛᵒ = 0 by definition.
    """
    model, alphabet, batch_converter = _load_model()

    wt_aa = sequence[position]

    # Build masked sequence
    masked_seq = sequence[:position] + "<mask>" + sequence[position + 1:]
    data = [("protein", masked_seq)]

    batch_labels, batch_strs, batch_tokens = batch_converter(data)

    # Forward pass (no grad needed)
    with torch.no_grad():
        logits = model(batch_tokens, repr_layers=[])["logits"]
        # logits shape: (1, seq_len+2, vocab_size)  (+2 for BOS/EOS tokens)

    # Position in token space: +1 because of BOS token
    token_position = position + 1
    logits_at_pos = logits[0, token_position, :]   # (vocab_size,)

    # Retrieve logit for WT
    wt_idx    = alphabet.get_idx(wt_aa)
    wt_logit  = logits_at_pos[wt_idx].item()

    # Compute ΔΨᵉᵛᵒ for each standard AA
    delta_psi = {}
    for aa in AA_20:
        aa_idx    = alphabet.get_idx(aa)
        aa_logit  = logits_at_pos[aa_idx].item()
        # Sign convention: higher = less likely / more deleterious
        delta_psi[aa] = -(aa_logit - wt_logit)

    return delta_psi   # WT residue → 0.0


# ---------------------------------------------------------------------------
# Scan: all positions (or a subset)
# ---------------------------------------------------------------------------
def compute_evo_scores_dms(
    sequence: str,
    positions: list[int] | None = None,    # 0-based; None = all positions
) -> pd.DataFrame:
    """
    Runs ΔΨᵉᵛᵒ for every (position, mutation) pair and returns a DataFrame.

    Columns
    -------
    Position_0based, Position_1based, WT, Mutation, Label, delta_psi_evo

    The 1-based index aligns with PyRosetta Pose numbering when the
    sequence starts at residue 1 (no gaps, no insertions).
    """
    if positions is None:
        positions = list(range(len(sequence)))

    records = []
    n = len(positions)

    for i, pos in enumerate(positions):
        log.info("ESM-2 scoring position %d/%d  (seq idx %d, AA=%s)",
                 i + 1, n, pos, sequence[pos])

        delta_psi = compute_delta_psi_evo(sequence, pos)
        wt_aa = sequence[pos]

        for mut_aa, dpsi in delta_psi.items():
            records.append({
                "Position_0based":  pos,
                "Position_1based":  pos + 1,      # Pose numbering
                "WT":               wt_aa,
                "Mutation":         mut_aa,
                "Label":            f"{wt_aa}{pos + 1}{mut_aa}",
                "delta_psi_evo":    dpsi,
            })

    df = pd.DataFrame(records)
    log.info("ESM-2 scoring complete: %d entries.", len(df))
    return df


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def _parse_args():
    parser = argparse.ArgumentParser(
        description="Compute ESM-2 ΔΨᵉᵛᵒ scores for dark energy calculation.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--sequence", required=True,
                        help="Protein sequence (single-letter codes).")
    parser.add_argument("--positions", nargs="+", type=int, default=None,
                        metavar="N",
                        help=(
                            "0-based positions to scan (space-separated). "
                            "Default: all positions in the sequence."
                        ))
    parser.add_argument("--output", default="evo_scores.csv",
                        help="Output CSV path (default: evo_scores.csv).")
    parser.add_argument("--plot", action="store_true",
                        help="Generate visualisation plots after scoring.")
    parser.add_argument("--plot-prefix", default=None, metavar="PATH",
                        help="File path prefix for saved plots. Omit to show interactively.")
    return parser.parse_args()


def main():
    args = _parse_args()
    positions = args.positions

    df = compute_evo_scores_dms(args.sequence, positions)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info("Saved → %s", args.output)

    if args.plot:
        from dms_plots import plot_evo_scores
        log.info("Generating ESM-2 evolutionary score plot…")
        plot_evo_scores(df, output_prefix=args.plot_prefix)


if __name__ == "__main__":
    main()