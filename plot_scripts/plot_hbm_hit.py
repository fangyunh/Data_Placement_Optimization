#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

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
    "font.size": 9,
    "axes.labelsize": 10,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
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
    # ---------- NEW: dynamically size figure width to fit legend on one line ----------
    present = list(df["method_key"].unique())
    legend_labels = ["Model weight ratio"]  # keep only the weight legend

    legend_fontsize = 8
    avg_char_width_in = legend_fontsize * 0.6 / 72.0  # ~0.6em at given fontsize
    text_inches = sum(len(lbl) for lbl in legend_labels) * avg_char_width_in
    handle_gap_inches = 0.40 * len(legend_labels)       # room for handles & gaps
    fig_w = max(4.0, text_inches + handle_gap_inches)  # minimum 6in, else fit text
    # ----------------------------------------------------------------------------------

    fig, ax = plt.subplots(figsize=(fig_w, 3.2))

    x = range(len(df))
    bar_w = 0.38

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
        ax.bar(i, w, width=bar_w, color=WEIGHT_FACE, edgecolor=EDGE_COLOR,
               hatch=WEIGHT_HATCH, linewidth=0.6, zorder=2)
        ax.bar(i, r, bottom=w, width=bar_w, color=base_color, edgecolor=EDGE_COLOR,
               linewidth=0.6, zorder=2)
        ax.text(i, w + r + 0.8, f"{w + r:.1f}%", ha="center", va="bottom", fontsize=8)

    # Modify the method names to have line breaks
    method_labels = []
    for k in df["method_key"]:
        name = display_names.get(k, k.capitalize())
        if ' ' in name:
            # Split at the first space and join with newline
            words = name.split(' ', 1)
            name = f"{words[0]}\n{words[1]}"
        method_labels.append(name)
        
    ax.set_xticks(list(x))
    # ax.set_xticklabels([display_names.get(k, k.capitalize()) for k in df["method_key"]],
    #                    rotation=0, fontsize=7)  # smaller x-tick labels
    ax.set_xticklabels(method_labels,
                       rotation=0, 
                       fontsize=7,
                       ha='center',  # Center align the text
                       va='top')  # Vertically center the text
    ax.set_ylabel("HBM hit rate (%)", fontsize=9)

    ymax = max(100.0, (df["hbm_hit_rate"].max() + 6))
    ax.set_ylim(0, min(100.0, ymax) if df["hbm_hit_rate"].max() <= 100 else ymax)

    ax.grid(axis="y", linestyle=":", linewidth=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    method_handles = []
    for m in PREFERRED_ORDER:
        if m in present:
            method_handles.append(Patch(facecolor=METHOD_COLORS[m], edgecolor=EDGE_COLOR,
                                        label=display_names.get(m, m.capitalize())))
    weight_handle = Patch(facecolor=WEIGHT_FACE, edgecolor=EDGE_COLOR, hatch=WEIGHT_HATCH,
                          label="Model weight")

    ax.legend(handles=[weight_handle],
              loc="upper center",
              frameon=False,
              bbox_to_anchor=(0.5, 1.12),
              ncol=1,          # keep single line
              fontsize=legend_fontsize,
              handlelength=1.4,
              borderaxespad=0.2)

    fig.tight_layout(pad=0.6)
    plt.subplots_adjust(bottom=0.25)
    fig.savefig(outfile_png, bbox_inches="tight")

    if clipped:
        print("[WARN] Some rows had Model weight ratio > HBM hit rate; clipped to bar height.", file=sys.stderr)

if __name__ == "__main__":
    df = load_and_prepare(CSV_PATH)
    if df.empty:
        raise SystemExit("No recognized methods found. Ensure 'methods' are among: "
                         + ", ".join(PREFERRED_ORDER))
    plot_hbm(df, outfile_png="hbm_hit_rate.png")
