# Variant Effect Predictor

Predicts whether a missense variant in a small cancer-gene panel (BRCA1, BRCA2, TP53,
PTEN, MLH1) is likely **Pathogenic** or **Benign**, trained on real ClinVar labels with
engineered physicochemical features - no LLM involved, classic supervised ML.

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -e ".[dev]"
```

## Pipeline

```bash
python -m variant_predictor.fetch_sequences   # pulls canonical UniProt sequences (~5s)
python -m variant_predictor.fetch_clinvar     # streams & filters ClinVar's 440MB file (~few min)
python -m variant_predictor.build_dataset     # joins labels + sequences -> data/dataset.csv
python -m variant_predictor.train             # trains & evaluates the model
```

## Predict a variant

```bash
python -m variant_predictor.predict BRCA1 p.Val1736Ala
```

Outputs a prediction, probability, and the top SHAP-attributed features driving it.

## Architecture

1. **`fetch_clinvar.py`** - streams ClinVar's `variant_summary.txt.gz` (~440MB
   compressed / ~9M rows) and filters to our 5-gene panel on the fly, so the full file
   is never written to disk.
2. **`fetch_sequences.py`** - pulls canonical protein sequences from UniProt for each
   gene in the panel.
3. **`features.py`** - parses missense protein HGVS notation (e.g. `p.Val1746Ile`) out
   of ClinVar's `Name` field, and builds physicochemical features per substitution:
   Grantham distance, hydrophobicity/charge/weight deltas, and normalized position in
   the protein.
4. **`build_dataset.py`** - joins ClinVar labels with the reference sequence, verifies
   the reference amino acid actually matches at that position (drops mismatches rather
   than risk a mislabeled row), and keeps only clear-cut Pathogenic/Benign calls
   (drops "Uncertain significance").
5. **`train.py`** - `HistGradientBoostingClassifier` (scikit-learn's built-in gradient
   boosting, no extra native dependency) trained on the engineered features plus a
   one-hot gene identity feature, evaluated against a naive majority-class baseline.
6. **`predict.py`** - CLI for a single variant, with a SHAP explanation of which
   features drove the prediction.

## Results (last run)

- **2,384** labeled missense variants across the 5-gene panel (1,370 benign / 1,014
  pathogenic)
- **AUC-ROC: 0.977**, accuracy 92% on a held-out 20% test split
- Naive baseline (always predict majority class): 57.4% accuracy

## Known limitations

- **Gene identity is doing real work.** The one-hot gene feature captures each gene's
  baseline pathogenic rate in ClinVar, which inflates apparent performance - this
  model is learning "which gene, plus how disruptive is the substitution," not a
  gene-agnostic notion of pathogenicity. A fairer eval would hold out an entire gene
  and test generalization to it, which this version doesn't do.
- **No conservation or structural features.** Evolutionary conservation and 3D
  structural context (e.g., is this residue buried, in a binding interface) are some
  of the most predictive real signals in this field and aren't included here - would
  require an MSA tool or structure-prediction integration.
- **No population frequency (gnomAD).** Rare variants skew pathogenic; this is a
  strong, easy-to-add signal left out of v1.
- **ClinVar label bias.** Better-studied genes and variants are overrepresented;
  ClinVar submissions aren't a random sample of the mutational landscape.
- **Not a clinical tool.** This is a portfolio/learning project, not validated for any
  real diagnostic use.
