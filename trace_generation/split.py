#!/usr/bin/env python3
"""
sparsify_attention_csv.py
────────────────────────────────────────────────────────────────────────────
Read a CSV that contains, per row:

    position,layer,indices_desc   # (or "scores", "read_kv" – handled flexibly)

`indices_desc` is a JSON list of token indices sorted from highest-attention
to lowest.  For each sparsity value *s* (e.g. 20 → drop the bottom 20 %),
we keep only the top (100-s) % of indices and write a new CSV:

    n,l,read_kv

One output file per sparsity:
    <input_basename>_<s>.csv
────────────────────────────────────────────────────────────────────────────
Usage example
─────────────
    sparsities = [20, 60, 85, 90]
    input_csv  = "qasper_abc123.csv"
    sparsify_csv(input_csv, sparsities)
"""

import csv, json, math, os, sys
from typing import List, Dict, IO

csv.field_size_limit(sys.maxsize) 

# ──────────────────────────────────────────────────────────────────────────
def _open_writers(input_path: str, sparsities: List[int]) -> Dict[int, IO]:
    """Create one CSV writer per sparsity value."""
    base, _ = os.path.splitext(os.path.basename(input_path))
    writers = {}
    for s in sparsities:
        out_path = f"{base}_{s}.csv"
        fh = open(out_path, "w", newline="")
        writer = csv.writer(fh)
        writer.writerow(["n", "l", "read_kv"])  # header
        writers[s] = (writer, fh)
    return writers

# ──────────────────────────────────────────────────────────────────────────
def _parse_index_col(row: dict) -> List[int]:
    """Robustly grab whichever column stores the index list."""
    for key in ("indices_desc", "scores", "read_kv"):
        if key in row:
            return json.loads(row[key])
    raise KeyError("No index column found in CSV row.")

# ──────────────────────────────────────────────────────────────────────────
def sparsify_csv(input_csv: str, sparsities: List[int]) -> None:
    """
    Parameters
    ----------
    input_csv   : str
        Path to the source CSV produced by your attention-logging script
        (one row per token × layer, JSON list of indices in the last column).
    sparsities  : List[int]
        Each integer *s* means *drop* the bottom *s* % of indices and keep
        the top (100-s) %.
    """
    if not sparsities:
        raise ValueError("sparsities list cannot be empty.")

    writers = _open_writers(input_csv, sparsities)

    with open(input_csv, newline="") as fin:
        reader = csv.DictReader(fin)
        for row in reader:
            position = row.get("position") or row.get("n")
            layer    = row.get("layer")    or row.get("l")
            indices  = _parse_index_col(row)
            L        = len(indices)

            for s in sparsities:
                keep_frac = 1.0 - s / 100.0
                k = max(1, int(math.ceil(L * keep_frac)))
                kept = indices[:k]

                writer, _fh = writers[s]
                writer.writerow([position, layer, json.dumps(kept)])

    # close files
    for _, fh in writers.values():
        fh.close()

    print(f"Finished: generated {len(sparsities)} sparsified CSV file(s).")

# ──────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # ── simple CLI for convenience ──
    if len(sys.argv) < 3:
        print(
            "USAGE: python sparsify_attention_csv.py <input_csv> <s1> [<s2> ...]",
            file=sys.stderr,
        )
        sys.exit(1)

    in_csv = sys.argv[1]
    spars   = [int(x) for x in sys.argv[2:]]
    sparsify_csv(in_csv, spars)
