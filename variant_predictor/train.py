"""Trains a gradient-boosted tree classifier on the engineered variant features
and evaluates it against a naive baseline.
"""
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"

FEATURE_COLS = [
    "grantham", "hydrophobicity_delta", "weight_delta", "charge_delta",
    "ref_hydrophobicity", "alt_hydrophobicity", "position_fraction",
]


def train(dataset_path: Path = None):
    dataset_path = dataset_path or DATA_DIR / "dataset.csv"
    MODELS_DIR.mkdir(exist_ok=True)

    df = pd.read_csv(dataset_path)
    df = pd.get_dummies(df, columns=["gene"], prefix="gene")
    gene_cols = [c for c in df.columns if c.startswith("gene_")]
    feature_cols = FEATURE_COLS + gene_cols

    X = df[feature_cols]
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = HistGradientBoostingClassifier(random_state=42)
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = model.predict(X_test)

    print("=== Model performance ===")
    print(f"AUC-ROC: {roc_auc_score(y_test, proba):.3f}")
    print(classification_report(y_test, preds, target_names=["Benign", "Pathogenic"]))

    baseline_preds = [y_train.mode()[0]] * len(y_test)
    baseline_acc = (pd.Series(baseline_preds).reset_index(drop=True) == y_test.reset_index(drop=True)).mean()
    print(f"Naive baseline (always predict majority class): {baseline_acc:.3f} accuracy")

    joblib.dump({"model": model, "feature_cols": feature_cols}, MODELS_DIR / "variant_model.joblib")
    print(f"Model saved to {MODELS_DIR / 'variant_model.joblib'}")
    return model


if __name__ == "__main__":
    train()
