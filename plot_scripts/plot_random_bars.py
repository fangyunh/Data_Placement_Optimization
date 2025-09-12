
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_latency_data(csv_path: str, target_randoms: list, target_methods: list) -> dict:
    """
    Reads a CSV with columns ['random','method','latency'] and returns
    a dict-of-lists keyed by random ratio and each method.
    Preserves order specified in target_randoms.
    """
    df = pd.read_csv(csv_path)
    # Filter data
    df = df[df['random'].isin(target_randoms) & df['method'].isin(target_methods)]
    
    # Create categorical type with specified order
    df['random'] = pd.Categorical(df['random'], categories=target_randoms, ordered=True)
    
    # Sort by the ordered categorical
    df = df.sort_values('random')
    
    pivot = df.pivot(index='random', columns='method', values='latency')
    # Important: preserve target_randoms order
    data = {'random': target_randoms}
    for method in pivot.columns:
        # Reorder the values according to tar_random
        values = []
        for r in target_randoms:
            values.append(pivot.loc[r, method])
        data[method] = values
    return data

def piecewise_scale(x, 
                    lower_src=(0.0, 1.0),
                    lower_dst=(0.0, 0.2),
                    upper_src=(1.0, None),
                    upper_shift=None):
    """
    Compresses [0, lower_src[1]] into [lower_dst[0], lower_dst[1]],
    then shifts x > lower_src[1] by the same amount (for continuity).
    """
    lo0, lo1 = lower_src
    dst0, dst1 = lower_dst
    if x <= lo1:
        return ((x - lo0) / (lo1 - lo0)) * (dst1 - dst0) + dst0
    # shift the upper region by (dst1 - lo1) so that at x=lo1 they meet
    shift = (dst1 - dst0) - (lo1 - lo0) if upper_shift is None else upper_shift
    return x + shift

if __name__ == "__main__":
    # --- 1) Load your data ----------------------------
    csv_path = "data/dif_random_60.csv"

    tar_random = ['low', 'actual', 'high']
    tar_methods = ['baseline', 'reuse', 'page', 'sa', 'best']

    # Add display name mapping
    display_names = {
        'baseline': 'Static Placement',
        'reuse': 'Reactive Scheduling',
        'sa': 'SA-Guided Scheduling',
        'page': 'Page Granularity Scheduling',
        'best': 'Best Case',
    }

    data = load_latency_data(csv_path, tar_random, tar_methods)
    randoms = data['random']                   # e.g. [85, 90, …]
    methods = [m for m in data.keys() if m != 'random']
    
    # --- 2) Compute “real” tokens decoded per run -------
    decode_token = 41538 - 31299   # e.g. 1024 * 12
    real_decode = decode_token * 0.4
    
    # --- 3) Tokens per second --------------------------
    tokens_per_sec = {}
    for m in methods:
        tokens_per_sec[m] = [
            real_decode / data[m][i]
            for i in range(len(randoms))
        ]
    
    # --- 4) Normalize (Best → 1.0) -----------------
    best_case = tokens_per_sec.get('best') or tokens_per_sec.get('Best')
    if best_case is None:
        raise ValueError("No 'best' column found for normalization")
    
    norm = {}
    for m in methods:
        if m.lower() == 'best':
            norm[m] = [1.0] * len(randoms)  # Best case is always 1.0
        else:
            norm[m] = [
                tokens_per_sec[m][i] / best_case[i]  # Divide by best value at each point
                for i in range(len(randoms))
            ]
    
    # --- 5) Piecewise scaling setup -------------------
    scaled = norm
    
    # --- 6) Plotting -----------------------------------
    plt.figure(figsize=(5.5, 5), dpi=300)
    ax = plt.gca()

    # Define plotting order and filter out 'best' (we'll plot normalized to it)
    plot_order = [m for m in tar_methods if m.lower() != 'best']
    n_methods = len(plot_order)
    n_groups = len(randoms)

    # === New: geometry on a unit grid ===================
    # Knobs you can tune:
    BAR_W_FRAC = 0.03        # bar thinness as fraction of unit group spacing (smaller -> thinner)
    GROUP_GAP_FRAC = 0.6    # gap between groups as fraction of unit spacing (smaller -> groups closer)

    group_spacing = 1.0                              # centers at 0,1,2,...
    group_w = (1.0 - GROUP_GAP_FRAC) * group_spacing # total width occupied by the bars in a group
    bar_w = BAR_W_FRAC * group_spacing + 0.05            # absolute bar width (in data units of the unit grid)

    # intra-group gap is computed so group width stays fixed regardless of bar_w
    if n_methods > 1:
        intra_gap = max(0.0, (group_w - n_methods * bar_w) / (n_methods - 1))
    else:
        intra_gap = 0.0

    # group centers and bar offsets
    index = np.arange(n_groups) * group_spacing
    offsets = (-group_w / 2 + bar_w / 2) + np.arange(n_methods) * (bar_w + intra_gap)

    # Freeze x-limits to the unit grid so autoscale can't fatten bars when gaps change
    ax.set_xlim(index[0] - group_spacing / 2, index[-1] + group_spacing / 2)
    ax.set_xticks(index)
    ax.set_xticklabels(tar_random, fontsize=14)
    ax.margins(x=0)  # no extra padding

    # choose colors
    colors = {
        'baseline': '#5a6d8c',
        'reuse': '#baccd9',
        'page': '#5697c3',
        'sa': '#11659a',
        'best': '#5a6d8c',
    }

    # display names (make sure keys match exactly)
    display_names = {
        'baseline': 'Static',
        'reuse': 'Reactive',
        'sa': 'SA-Guided',
        'page': 'Page-Granularity',
        'best': 'Best Case',
    }

    # Plot bars (skip 'best')
    for j, m in enumerate(plot_order):
        xs = index + offsets[j]
        label = display_names.get(m, m)
        ax.bar(
            xs,
            norm[m],           # already normalized to 'best'
            width=bar_w,
            label=label,
            color=colors[m.lower()],
            zorder=2
        )

    # axes labels & ticks
    ax.set_xlabel('Token Importance Variation', fontsize=16)
    ax.set_ylabel('Normalized tokens/sec', fontsize=16)
    ax.set_yticks(np.arange(0, 1.1, 0.1))
    ax.set_yticklabels([f"{t:.1f}" for t in np.arange(0, 1.1, 0.1)], fontsize=12)
    ax.set_ylim(0, 1.0)

    ax.grid(axis='y', linestyle='--', alpha=0.7, zorder=1)

    # compact, one-row legend above plot
    ax.legend(
        loc='upper center',
        bbox_to_anchor=(0.5, 1.15),
        ncol=min(5, n_methods),
        fontsize=10,
        frameon=False,
        borderaxespad=0.
    )

    plt.tight_layout()
    plt.savefig("random.png", dpi=300, bbox_inches="tight")
    plt.show()
