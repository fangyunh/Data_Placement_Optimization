#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np  # NEW: needed for compact x positions

CSV_PATH = "data/hbm_hit_rate_60.csv"

PREFERRED_ORDER = ["baseline", "reuse", "lookahead", "page", "sa", "best"]

METHOD_COLORS = {
    "baseline": "#5a6d8c",
    "reuse":    "#baccd9",
    "page":     "#5697c3",
    "sa":       "#11659a",
    "best":     "#126d82",
    "lookahead":"#7fb3d5",
}

display_names = {
    'baseline': 'Static Placement',
    'reuse': 'Reactive Scheduling',
    'sa': 'SA-Guided Scheduling',
    'page': 'Page-Granularity Scheduling',
    'best': 'Unlimited HBM',
    'lookahead': 'Lookahead Scheduling',
}

WEIGHT_FACE = "#d9e6f2"
WEIGHT_HATCH = "///"
EDGE_COLOR = "black"

plt.rcParams.update({
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.family": "DejaVu Serif",
    "font.size": 8,
    "axes.labelsize": 9,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
    "axes.xmargin": 0.01,  # NEW: reduce default side padding
})

def _to_percent_series(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().max() <= 1.5:
        return s * 100.0
    return s

def load_and_prepare(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    lower = {c.lower().strip(): c for c in df.columns}
    try:
        mcol = lower[[c for c in lower if c.startswith("methods") or c == "method"][0]]
        hcol = lower[[c for c in lower if "hbm" in c and "hit" in c][0]]
        wcol = lower[[c for c in lower if "model" in c and "ratio" in c][0]]
    except IndexError:
        raise ValueError("Required columns not found. Need: methods, HBM hit rate, Model weight ratio")

    out = pd.DataFrame({
        "method": df[mcol].astype(str).str.strip(),
        "hbm_hit_rate": _to_percent_series(df[hcol]),
        "weight_ratio": _to_percent_series(df[wcol]),
    })
    out["method_key"] = out["method"].str.lower()
    order_map = {m: i for i, m in enumerate(PREFERRED_ORDER)}
    out["order_key"] = out["method_key"].map(order_map)
    out = (out.dropna(subset=["order_key"])
              .sort_values("order_key")
              .drop(columns=["order_key"])
              .reset_index(drop=True))
    return out

def plot_hbm(df: pd.DataFrame, outfile_png: str = "hbm_hit_rate.png"):
    # ---------- Figure width heuristic (kept) ----------
    legend_labels = ["model weight ratio"]
    legend_fontsize = 7
    avg_char_width_in = legend_fontsize * 0.6 / 72.0
    text_inches = sum(len(lbl) for lbl in legend_labels) * avg_char_width_in
    handle_gap_inches = 0.40 * len(legend_labels)
    fig_w = max(3.0, text_inches + handle_gap_inches)
    # ---------------------------------------------------

    fig, ax = plt.subplots(figsize=(fig_w, 1.6))

    # ---- NEW: compact spacing between bar centers ----
    BAR_SPACING = 0.3          # < 1.0 packs bars closer than default
    BAR_WIDTH   = 0.18
    x = np.arange(len(df)) * BAR_SPACING
    bar_w = BAR_SPACING * 0.85  # wide bars relative to spacing
    # --------------------------------------------------

    clipped = False
    weight_portion, remainder_portion = [], []
    for hr, wr in zip(df["hbm_hit_rate"].values, df["weight_ratio"].values):
        w = min(max(wr, 0.0), hr)
        if wr > hr:
            clipped = True
        weight_portion.append(w)
        remainder_portion.append(hr - w)

    for i, (m_key, w, r) in enumerate(zip(df["method_key"], weight_portion, remainder_portion)):
        base_color = METHOD_COLORS.get(m_key, "#9e9e9e")
        ax.bar(x[i], w, width=BAR_WIDTH, color=WEIGHT_FACE, edgecolor=EDGE_COLOR,
               hatch=WEIGHT_HATCH, linewidth=0.6, zorder=2)
        ax.bar(x[i], r, bottom=w, width=BAR_WIDTH, color=base_color, edgecolor=EDGE_COLOR,
               linewidth=0.6, zorder=2)
        ax.text(x[i], w + r + 0.8, f"{w + r:.1f}%", ha="center", va="bottom", fontsize=5)

    # Modified method labels (kept)
    method_labels = []
    for k in df["method_key"]:
        name = display_names.get(k, k.capitalize())
        if ' ' in name:
            words = name.split(' ', 1)
            name = f"{words[0]}\n{words[1]}"
        method_labels.append(name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(method_labels, rotation=0, fontsize=5, ha='center', va='top')
    ax.set_ylabel("HBM hit rate (%)", fontsize=9)

    ymax = max(100.0, (df["hbm_hit_rate"].max() + 6))
    ax.set_ylim(0, min(100.0, ymax) if df["hbm_hit_rate"].max() <= 100 else ymax)

    # ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # NEW: tighten side limits so bars sit snugly inside the frame
    if len(x) > 0:
        pad = max(0.005, 0.5 * (BAR_SPACING - BAR_WIDTH))
        ax.set_xlim(x[0] - BAR_WIDTH/2 - pad, x[-1] + BAR_WIDTH/2 + pad)
    ax.margins(x=0.01)

    # Legend (kept to only weight handle)
    weight_handle = Patch(facecolor=WEIGHT_FACE, edgecolor=EDGE_COLOR, hatch=WEIGHT_HATCH,
                          label="model weight ratio")
    ax.legend(handles=[weight_handle],
              loc="upper center",
              frameon=False,
              bbox_to_anchor=(0.5, 1.12),
              ncol=1,
              fontsize=legend_fontsize,
              handlelength=1.4,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.6)
    plt.subplots_adjust(bottom=0.22)
    fig.savefig(outfile_png, bbox_inches="tight")

    if clipped:
        print("[WARN] Some rows had Model weight ratio > HBM hit rate; clipped to bar height.", file=sys.stderr)

if __name__ == "__main__":
    df = load_and_prepare(CSV_PATH)
    if df.empty:
        raise SystemExit("No recognized methods found. Ensure 'methods' are among: "
                         + ", ".join(PREFERRED_ORDER))
    plot_hbm(df, outfile_png="hbm_hit_rate.png")
