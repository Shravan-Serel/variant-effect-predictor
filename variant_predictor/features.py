"""Feature engineering: parses protein HGVS notation out of ClinVar's Name field and
builds physicochemical features for each missense substitution.
"""
import re

# Three-letter -> one-letter amino acid codes
AA_3TO1 = {
    "Ala": "A", "Arg": "R", "Asn": "N", "Asp": "D", "Cys": "C",
    "Gln": "Q", "Glu": "E", "Gly": "G", "His": "H", "Ile": "I",
    "Leu": "L", "Lys": "K", "Met": "M", "Phe": "F", "Pro": "P",
    "Ser": "S", "Thr": "T", "Trp": "W", "Tyr": "Y", "Val": "V",
}

# Kyte-Doolittle hydrophobicity scale
HYDROPHOBICITY = {
    "A": 1.8, "R": -4.5, "N": -3.5, "D": -3.5, "C": 2.5,
    "Q": -3.5, "E": -3.5, "G": -0.4, "H": -3.2, "I": 4.5,
    "L": 3.8, "K": -3.9, "M": 1.9, "F": 2.8, "P": -1.6,
    "S": -0.8, "T": -0.7, "W": -0.9, "Y": -1.3, "V": 4.2,
}

# Molecular weight (Da) of each amino acid residue
WEIGHT = {
    "A": 89, "R": 174, "N": 132, "D": 133, "C": 121,
    "Q": 146, "E": 147, "G": 75, "H": 155, "I": 131,
    "L": 131, "K": 146, "M": 149, "F": 165, "P": 115,
    "S": 105, "T": 119, "W": 204, "Y": 181, "V": 117,
}

# Charge at physiological pH: -1, 0, or 1
CHARGE = {
    "A": 0, "R": 1, "N": 0, "D": -1, "C": 0,
    "Q": 0, "E": -1, "G": 0, "H": 0, "I": 0,
    "L": 0, "K": 1, "M": 0, "F": 0, "P": 0,
    "S": 0, "T": 0, "W": 0, "Y": 0, "V": 0,
}

# Grantham distance matrix (physicochemical distance between amino acid pairs).
# Sourced from Grantham (1974) - reduced here to a lookup built from composition,
# polarity and volume differences per residue as a practical approximation.
_GRANTHAM_COMPOSITION = {
    "A": (0.0, 8.1, 31), "R": (0.65, 10.5, 124), "N": (1.33, 11.6, 56),
    "D": (1.38, 13.0, 54), "C": (2.75, 5.5, 55), "Q": (0.89, 10.5, 85),
    "E": (0.92, 12.3, 83), "G": (0.74, 9.0, 3), "H": (0.58, 10.4, 96),
    "I": (0.0, 5.2, 111), "L": (0.0, 4.9, 111), "K": (0.33, 11.3, 119),
    "M": (0.0, 5.7, 105), "F": (0.0, 5.2, 132), "P": (0.39, 8.0, 32.5),
    "S": (1.42, 9.2, 32), "T": (0.71, 8.6, 61), "W": (0.13, 5.4, 170),
    "Y": (0.20, 6.2, 136), "V": (0.0, 5.9, 84),
}
_C, _P, _V = 1.833, 0.1018, 0.000399


def grantham_distance(a: str, b: str) -> float:
    if a not in _GRANTHAM_COMPOSITION or b not in _GRANTHAM_COMPOSITION:
        return float("nan")
    c1, p1, v1 = _GRANTHAM_COMPOSITION[a]
    c2, p2, v2 = _GRANTHAM_COMPOSITION[b]
    return (_C * (c1 - c2) ** 2 + _P * (p1 - p2) ** 2 + _V * (v1 - v2) ** 2) ** 0.5


# Matches e.g. "(p.Val1746Ile)" inside a ClinVar Name field
_HGVS_P_RE = re.compile(r"p\.([A-Za-z]{3})(\d+)([A-Za-z]{3})\b")


def parse_missense(name: str):
    """Returns (ref_aa, pos, alt_aa) one-letter codes for a missense HGVS protein
    change, or None if the Name field isn't a simple missense substitution."""
    match = _HGVS_P_RE.search(name)
    if not match:
        return None
    ref3, pos, alt3 = match.groups()
    if ref3 in ("Ter",) or alt3 in ("Ter",):
        return None  # nonsense, not missense
    ref1 = AA_3TO1.get(ref3)
    alt1 = AA_3TO1.get(alt3)
    if not ref1 or not alt1 or ref1 == alt1:
        return None
    return ref1, int(pos), alt1


def build_features(gene: str, ref_aa: str, pos: int, alt_aa: str, protein_length: int) -> dict:
    return {
        "gene": gene,
        "grantham": grantham_distance(ref_aa, alt_aa),
        "hydrophobicity_delta": HYDROPHOBICITY[alt_aa] - HYDROPHOBICITY[ref_aa],
        "weight_delta": WEIGHT[alt_aa] - WEIGHT[ref_aa],
        "charge_delta": CHARGE[alt_aa] - CHARGE[ref_aa],
        "ref_hydrophobicity": HYDROPHOBICITY[ref_aa],
        "alt_hydrophobicity": HYDROPHOBICITY[alt_aa],
        "position_fraction": pos / protein_length if protein_length else float("nan"),
    }
