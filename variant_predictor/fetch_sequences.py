"""Downloads canonical protein sequences (FASTA) from UniProt for the gene panel."""
import json
from pathlib import Path

import requests

from variant_predictor.genes import GENE_PANEL

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "sequences.json"


def fetch() -> dict:
    DATA_DIR.mkdir(exist_ok=True)
    sequences = {}
    for gene, accession in GENE_PANEL.items():
        url = f"https://rest.uniprot.org/uniprotkb/{accession}.fasta"
        resp = requests.get(url, timeout=30)
        resp.raise_for_status()
        lines = resp.text.strip().split("\n")
        seq = "".join(lines[1:])
        sequences[gene] = seq
        print(f"{gene} ({accession}): {len(seq)} aa")

    OUT_PATH.write_text(json.dumps(sequences, indent=2), encoding="utf-8")
    return sequences


if __name__ == "__main__":
    fetch()
