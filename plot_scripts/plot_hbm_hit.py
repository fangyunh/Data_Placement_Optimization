#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import numpy as np

# =========================
# CONFIG — Tweak these only
# =========================
CSV_PATH        = "data/hbm_hit_rate_60.csv"  # input CSV path
OUTFILE_PNG     = "hbm_hit_rate.png"          # output image

# Geometry knobs (independent):
CENTER_SPACING  = 0.20     # distance between bar centers (smaller => bars closer)
BAR_WIDTH_FRAC  = 0.30     # bar width as a fraction of CENTER_SPACING (0..1)
BAR_WIDTH_ABS   = 0.08    # absolute bar width in data units; overrides BAR_WIDTH_FRAC if not None
SIDE_PAD_FRAC   = 0.1    # small padding on left/right sides as fraction of CENTER_SPACING

# Figure size:
FIG_W           = 3.2      # inches
FIG_H           = 3      # inches

# Labels and legend:
SHOW_PERCENT_LABELS = True
PERCENT_FONT_SIZE   = 6
LEGEND_MODE         = "top"  # "right", "top", or "none"
LEGEND_FONT_SIZE    = 8

# Aesthetics:
WEIGHT_FACE   = "#d9e6f2"
WEIGHT_HATCH  = "///"
EDGE_COLOR    = "black"

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
    "axes.xmargin": 0.01,
})

# Ordering and colors
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


# =========================
# Helpers
# =========================
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

    # Keep only preferred methods in the requested order
    order_map = {m: i for i, m in enumerate(PREFERRED_ORDER)}
    out["order_key"] = out["method_key"].map(order_map)
    out = (out.dropna(subset=["order_key"])
              .sort_values("order_key")
              .drop(columns=["order_key"])
              .reset_index(drop=True))
    return out


# =========================
# Plot
# =========================
def plot_hbm(df: pd.DataFrame, outfile_png: str = OUTFILE_PNG):
    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))

    # --- Geometry: place bars on a unit-like grid controlled by CENTER_SPACING
    x = np.arange(len(df)) * float(CENTER_SPACING)

    if BAR_WIDTH_ABS is not None:
        bar_width = float(BAR_WIDTH_ABS)
    else:
        bar_width = float(BAR_WIDTH_FRAC) * float(CENTER_SPACING)
    bar_width = max(1e-6, bar_width)  # avoid zero/negative widths

    # Split bars into hatched "weight" portion + remainder of HBM hit rate
    clipped = False
    weight_portion, remainder_portion = [], []
    for hr, wr in zip(df["hbm_hit_rate"].values, df["weight_ratio"].values):
        w = min(max(wr, 0.0), hr)
        if wr > hr:
            clipped = True
        weight_portion.append(w)
        remainder_portion.append(hr - w)

    for xi, m_key, w, r in zip(x, df["method_key"], weight_portion, remainder_portion):
        base_color = METHOD_COLORS.get(m_key, "#9e9e9e")
        ax.bar(xi, w, width=bar_width, color=WEIGHT_FACE, edgecolor=EDGE_COLOR,
               hatch=WEIGHT_HATCH, linewidth=0.6, zorder=2)
        ax.bar(xi, r, bottom=w, width=bar_width, color=base_color, edgecolor=EDGE_COLOR,
               linewidth=0.6, zorder=2)
        if SHOW_PERCENT_LABELS:
            ax.text(xi, w + r + 0.8, f"{w + r:.1f}%", ha="center", va="bottom",
                    fontsize=PERCENT_FONT_SIZE)

    # X tick labels (two-line where applicable)
    method_labels = []
    for k in df["method_key"]:
        name = display_names.get(k, k.capitalize())
        if ' ' in name:
            a, b = name.split(' ', 1)
            name = f"{a}\n{b}"
        method_labels.append(name)

    ax.set_xticks(list(x))
    ax.set_xticklabels(method_labels, rotation=0, fontsize=7, ha='center', va='top')

    # Y axis
    ymax = max(100.0, (df["hbm_hit_rate"].max() + 6))
    ax.set_ylim(0, min(100.0, ymax) if df["hbm_hit_rate"].max() <= 100 else ymax)
    ax.set_ylabel("HBM hit rate (%)", fontsize=9)

    # Cosmetics
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Tight x-limits so bars sit snugly inside the frame; small side padding
    if len(x) > 0:
        pad = max(0.0, SIDE_PAD_FRAC * float(CENTER_SPACING))
        ax.set_xlim(x[0] - bar_width/2 - pad, x[-1] + bar_width/2 + pad)
    ax.margins(x=0)

    # Legend — only the weight hatch handle, placed per LEGEND_MODE
    if LEGEND_MODE.lower() != "none":
        weight_handle = Patch(facecolor=WEIGHT_FACE, edgecolor=EDGE_COLOR,
                              hatch=WEIGHT_HATCH, label="model weight ratio")
        if LEGEND_MODE.lower() == "right":
            ax.legend(
                handles=[weight_handle],
                loc="center left",
                bbox_to_anchor=(1.02, 0.5),  # just outside the axes on the right
                frameon=False,
                ncol=1,
                fontsize=LEGEND_FONT_SIZE,
                handlelength=1.4,
                borderaxespad=0.0,
            )
        elif LEGEND_MODE.lower() == "top":
            ax.legend(
                handles=[weight_handle],
                loc="upper center",
                bbox_to_anchor=(0.5, 1.10),
                frameon=False,
                ncol=1,
                fontsize=LEGEND_FONT_SIZE,
                handlelength=1.4,
                borderaxespad=0.2,
            )

    fig.tight_layout(pad=0.6)
    # bbox_inches="tight" ensures right-side legend (if any) is fully included
    fig.savefig(outfile_png, bbox_inches="tight")

    if clipped:
        print("[WARN] Some rows had Model weight ratio > HBM hit rate; clipped to bar height.",
              file=sys.stderr)


# =========================
# Main
# =========================
if __name__ == "__main__":
    df = load_and_prepare(CSV_PATH)
    if df.empty:
        raise SystemExit(
            "No recognized methods found. Ensure 'methods' are among: " + ", ".join(PREFERRED_ORDER)
        )
    plot_hbm(df, outfile_png=OUTFILE_PNG)
