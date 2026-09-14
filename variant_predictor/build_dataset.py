"""Joins the filtered ClinVar rows with UniProt sequences to build a labeled
feature table for missense variants only, keeping just clear-cut Pathogenic /
Benign labels (dropping Uncertain significance and non-missense variants).
"""
import json
from pathlib import Path

import pandas as pd

from variant_predictor.features import build_features, parse_missense

DATA_DIR = Path(__file__).resolve().parent.parent / "data"

PATHOGENIC_LABELS = {"Pathogenic", "Likely pathogenic", "Pathogenic/Likely pathogenic"}
BENIGN_LABELS = {"Benign", "Likely benign", "Benign/Likely benign"}


def build(clinvar_path: Path = None, sequences_path: Path = None, out_path: Path = None) -> pd.DataFrame:
    clinvar_path = clinvar_path or DATA_DIR / "clinvar_filtered.csv"
    sequences_path = sequences_path or DATA_DIR / "sequences.json"
    out_path = out_path or DATA_DIR / "dataset.csv"

    sequences = json.loads(sequences_path.read_text(encoding="utf-8"))
    df = pd.read_csv(clinvar_path, low_memory=False)

    rows = []
    for _, r in df.iterrows():
        sig = str(r.get("ClinicalSignificance", ""))
        if sig in PATHOGENIC_LABELS:
            label = 1
        elif sig in BENIGN_LABELS:
            label = 0
        else:
            continue

        gene = r.get("GeneSymbol")
        name = str(r.get("Name", ""))
        parsed = parse_missense(name)
        if parsed is None or gene not in sequences:
            continue
        ref_aa, pos, alt_aa = parsed

        seq = sequences[gene]
        if pos < 1 or pos > len(seq) or seq[pos - 1] != ref_aa:
            continue  # reference mismatch - skip rather than risk a mislabeled row

        feats = build_features(gene, ref_aa, pos, alt_aa, len(seq))
        feats["label"] = label
        feats["variant_id"] = f"{gene}:p.{ref_aa}{pos}{alt_aa}"
        rows.append(feats)

    out_df = pd.DataFrame(rows).drop_duplicates(subset="variant_id")
    out_df.to_csv(out_path, index=False)
    print(f"Built dataset: {len(out_df)} labeled variants -> {out_path}")
    print(out_df["label"].value_counts())
    print(out_df["gene"].value_counts())
    return out_df


if __name__ == "__main__":
    build()
