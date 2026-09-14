"""Streams the ClinVar variant_summary.txt.gz file and keeps only rows for our gene
panel, so we never have to store the full ~440MB (compressed) / ~3GB (uncompressed)
file on disk.
"""
import csv
import gzip
import sys
from pathlib import Path

import requests

from variant_predictor.genes import GENE_PANEL

CLINVAR_URL = "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/tab_delimited/variant_summary.txt.gz"
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
OUT_PATH = DATA_DIR / "clinvar_filtered.csv"


def fetch(genes=None, out_path: Path = OUT_PATH) -> int:
    genes = set(genes or GENE_PANEL.keys())
    DATA_DIR.mkdir(exist_ok=True)

    print(f"Streaming {CLINVAR_URL} ...")
    resp = requests.get(CLINVAR_URL, stream=True, timeout=60)
    resp.raise_for_status()

    kept = 0
    scanned = 0
    with gzip.GzipFile(fileobj=resp.raw) as gz, open(out_path, "w", newline="", encoding="utf-8") as out:
        reader = csv.reader((line.decode("utf-8", errors="replace") for line in gz), delimiter="\t")
        header = next(reader)
        gene_idx = header.index("GeneSymbol")
        writer = csv.writer(out)
        writer.writerow(header)

        for row in reader:
            scanned += 1
            if scanned % 500_000 == 0:
                print(f"  scanned {scanned:,} rows, kept {kept:,}")
            if row[gene_idx] in genes:
                writer.writerow(row)
                kept += 1

    print(f"Done. Scanned {scanned:,} rows, kept {kept:,} for genes {sorted(genes)}.")
    return kept


if __name__ == "__main__":
    fetch()
