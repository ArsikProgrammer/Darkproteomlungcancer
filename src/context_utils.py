"""Shared helpers for trinucleotide context extraction."""
from pathlib import Path


COMP = str.maketrans("ACGT", "TGCA")
COMP_MAP = {"A": "T", "T": "A", "C": "G", "G": "C"}


def rc(s: str) -> str:
    return s.translate(COMP)[::-1]


def context_for_position(seq: str, pos: int, ref: str, alt: str):
    """Return (raw_trinucleotide, norm_trinucleotide, context_96) or None."""
    ref = ref.upper()
    alt = alt.upper()
    if pos - 2 < 0 or pos + 1 > len(seq):
        return None
    seq3 = seq[pos - 2:pos + 1].upper()
    if len(seq3) != 3 or seq3[1] != ref:
        return None
    tri = seq3
    r, a = ref, alt
    if ref in "AG":
        tri = rc(tri)
        r = COMP_MAP[ref]
        a = COMP_MAP[alt]
    label = f"{tri[0]}[{r}>{a}]{tri[2]}"
    return seq3, tri, label


def normalize_chrom(chrom: str) -> str:
    chrom = str(chrom).strip()
    if chrom.lower().startswith("chr"):
        chrom = chrom[3:]
    if chrom == "23":
        return "X"
    if chrom == "24":
        return "Y"
    return chrom

