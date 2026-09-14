import json
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from variant_predictor.genes import GENE_PANEL
from variant_predictor.predict import predict

DATA_DIR = Path(__file__).resolve().parent / "data"
MODELS_DIR = Path(__file__).resolve().parent / "models"

st.set_page_config(page_title="Variant Effect Predictor", page_icon="🧬", layout="centered")

st.title("🧬 Variant Effect Predictor")
st.caption(
    "Predicts whether a missense variant in a 5-gene cancer panel is likely "
    "Pathogenic or Benign. Trained on real ClinVar labels with engineered "
    "physicochemical features (Grantham distance, hydrophobicity, charge, position)."
)

if not (MODELS_DIR / "variant_model.joblib").exists():
    st.error(
        "No trained model found. Run the pipeline first:\n\n"
        "`python -m variant_predictor.fetch_sequences`\n"
        "`python -m variant_predictor.fetch_clinvar`\n"
        "`python -m variant_predictor.build_dataset`\n"
        "`python -m variant_predictor.train`"
    )
    st.stop()

sequences = json.loads((DATA_DIR / "sequences.json").read_text(encoding="utf-8"))

tab_predict, tab_dataset, tab_about = st.tabs(["Predict", "Training Data", "About"])

with tab_predict:
    col1, col2 = st.columns([1, 2])
    with col1:
        gene = st.selectbox("Gene", list(GENE_PANEL.keys()))
    with col2:
        example = {
            "BRCA1": "p.Val1736Ala",
            "BRCA2": "p.Asn372His",
            "TP53": "p.Arg175His",
            "PTEN": "p.Arg130Gln",
            "MLH1": "p.Ile219Val",
        }[gene]
        protein_change = st.text_input(
            "Protein change (HGVS)", value=example, help="e.g. p.Val1736Ala"
        )

    seq_len = len(sequences[gene])
    st.caption(f"{gene} canonical isoform: {seq_len} amino acids ({GENE_PANEL[gene]})")

    if st.button("Predict", type="primary"):
        try:
            result = predict(gene, protein_change)
        except ValueError as e:
            st.error(str(e))
        else:
            proba = result["pathogenic_probability"]
            label = result["prediction"]

            if label == "Pathogenic":
                st.error(f"### Prediction: {label}  (p = {proba:.3f})")
            else:
                st.success(f"### Prediction: {label}  (p = {proba:.3f})")

            st.progress(proba, text=f"Pathogenic probability: {proba:.1%}")

            st.subheader("Top factors driving this prediction")
            factors = result["top_factors"]
            fig = go.Figure(
                go.Bar(
                    x=[f["shap_contribution"] for f in factors],
                    y=[f["feature"] for f in factors],
                    orientation="h",
                    marker_color=[
                        "#d62728" if f["shap_contribution"] > 0 else "#2ca02c" for f in factors
                    ],
                )
            )
            fig.update_layout(
                xaxis_title="SHAP contribution (+ pushes toward Pathogenic)",
                yaxis=dict(autorange="reversed"),
                height=300,
                margin=dict(l=10, r=10, t=10, b=10),
            )
            st.plotly_chart(fig, use_container_width=True)
            st.json(result)

with tab_dataset:
    dataset_path = DATA_DIR / "dataset.csv"
    if dataset_path.exists():
        df = pd.read_csv(dataset_path)
        st.write(f"**{len(df)}** labeled missense variants across the gene panel")

        c1, c2 = st.columns(2)
        with c1:
            st.write("**By label**")
            st.bar_chart(df["label"].map({0: "Benign", 1: "Pathogenic"}).value_counts())
        with c2:
            st.write("**By gene**")
            st.bar_chart(df["gene"].value_counts())

        st.write("**Sample rows**")
        st.dataframe(df.sample(min(20, len(df))), use_container_width=True)
    else:
        st.info("Run `python -m variant_predictor.build_dataset` to generate the dataset.")

with tab_about:
    st.markdown(
        """
Built on ClinVar (NCBI) variant labels and UniProt canonical sequences for a 5-gene
cancer panel: **BRCA1, BRCA2, TP53, PTEN, MLH1**.

**Model**: scikit-learn `HistGradientBoostingClassifier` on engineered physicochemical
features (Grantham distance, hydrophobicity/charge/weight deltas, normalized position)
plus gene identity.

**Known limitations**
- Gene identity carries real predictive weight (each gene's baseline pathogenic rate
  in ClinVar), so this isn't a purely gene-agnostic pathogenicity score.
- No evolutionary conservation or structural features yet.
- No population allele frequency (gnomAD) yet.
- Not a clinical or diagnostic tool — a portfolio/learning project.

Source: [github.com/Shravan-Serel/variant-effect-predictor](https://github.com/Shravan-Serel/variant-effect-predictor)
"""
    )
