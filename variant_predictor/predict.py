"""CLI: predict pathogenicity for a single variant, e.g.
python -m variant_predictor.predict BRCA1 p.Val1746Ile
"""
import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import shap

from variant_predictor.features import build_features, parse_missense

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def predict(gene: str, protein_change: str) -> dict:
    bundle = joblib.load(MODELS_DIR / "variant_model.joblib")
    model, feature_cols = bundle["model"], bundle["feature_cols"]
    sequences = json.loads((DATA_DIR / "sequences.json").read_text(encoding="utf-8"))

    if gene not in sequences:
        raise ValueError(f"Gene {gene} is not in the trained panel: {list(sequences)}")
    seq = sequences[gene]

    parsed = parse_missense(f"(p.{protein_change.removeprefix('p.')})")
    if parsed is None:
        raise ValueError(f"Could not parse '{protein_change}' as a missense change (e.g. p.Val1746Ile)")
    ref_aa, pos, alt_aa = parsed
    if pos < 1 or pos > len(seq) or seq[pos - 1] != ref_aa:
        raise ValueError(
            f"Reference mismatch: {gene} position {pos} is '{seq[pos - 1] if 1 <= pos <= len(seq) else '?'}', not '{ref_aa}'"
        )

    feats = build_features(gene, ref_aa, pos, alt_aa, len(seq))
    row = pd.DataFrame([feats])
    row = pd.get_dummies(row, columns=["gene"], prefix="gene")
    for col in feature_cols:
        if col not in row.columns:
            row[col] = 0
    row = row[feature_cols]

    proba = model.predict_proba(row)[0, 1]
    label = "Pathogenic" if proba >= 0.5 else "Benign"

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(row)
    contributions = sorted(
        zip(feature_cols, shap_values[0]), key=lambda kv: abs(kv[1]), reverse=True
    )[:5]

    return {
        "variant": f"{gene}:p.{ref_aa}{pos}{alt_aa}",
        "prediction": label,
        "pathogenic_probability": round(float(proba), 3),
        "top_factors": [{"feature": f, "shap_contribution": round(float(v), 4)} for f, v in contributions],
    }


def main():
    parser = argparse.ArgumentParser(description="Predict variant pathogenicity")
    parser.add_argument("gene", help="Gene symbol, e.g. BRCA1")
    parser.add_argument("protein_change", help="Protein change, e.g. p.Val1746Ile or Val1746Ile")
    args = parser.parse_args()

    result = predict(args.gene, args.protein_change)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
