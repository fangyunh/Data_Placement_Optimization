
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

def load_latency_data(csv_path: str, target_sparsities: list, target_methods: list) -> dict:
    """
    Reads a CSV with columns ['sparsity','method','latency'] and returns
    a dict-of-lists keyed by sparsity and each method.
    Filters data based on specified sparsities and methods.
    """
    df = pd.read_csv(csv_path)
    # Filter data
    df = df[df['sparsity'].isin(target_sparsities) & df['method'].isin(target_methods)]
    pivot = df.pivot(index='sparsity', columns='method', values='latency')
    data = {'sparsity': pivot.index.tolist()}
    for method in pivot.columns:
        data[method] = pivot[method].tolist()
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
    csv_path = "data/latency_vs_sparsity.csv"

    tar_sparsities = [40, 60, 85, 90]
    tar_methods = ['baseline', 'reuse', 'sa']

    data = load_latency_data(csv_path, tar_sparsities, tar_methods)
    sparsities = data['sparsity']                   # e.g. [85, 90, …]
    methods = [m for m in data.keys() if m != 'sparsity']
    
    # --- 2) Compute “real” tokens decoded per run -------
    decode_token = 12_288  # e.g. 1024 * 12
    real_decode = [decode_token * (1 - s / 100.0) for s in sparsities]
    
    # --- 3) Tokens per second --------------------------
    tokens_per_sec = {}
    for m in methods:
        tokens_per_sec[m] = [
            real_decode[i] / data[m][i]
            for i in range(len(sparsities))
        ]
    
    # --- 4) Normalize (Baseline → 1.0) -----------------
    baseline = tokens_per_sec.get('baseline') or tokens_per_sec.get('Baseline')
    if baseline is None:
        raise ValueError("No 'baseline' column found for normalization")
    
    norm = {}
    for m in methods:
        if m.lower() == 'baseline':
            norm[m] = [1.0] * len(sparsities)
        else:
            norm[m] = [
                tokens_per_sec[m][i] / baseline[i]
                for i in range(len(sparsities))
            ]
    
    # --- 5) Piecewise scaling setup -------------------
    LOWER_DST = (0.0, 0.6)
    scaled = {
        m: [piecewise_scale(x,
                            lower_src=(0.0, 1.0),
                            lower_dst=LOWER_DST)
            for x in norm[m]]
        for m in methods
    }
    
    # --- 6) Plotting -----------------------------------
    plt.figure(figsize=(10, 4), dpi=300)
    ax = plt.gca()
    
    # bar geometry
    n_methods = len(methods) - 1
    bar_w = 0.8
    gap = 0.05
    group_gap = 1.0
    total_w = n_methods * bar_w + (n_methods - 1) * gap
    # x positions for each group
    index = np.arange(len(sparsities)) * (total_w + group_gap)
    # offsets to center each bar in a group
    offsets = (np.arange(n_methods) * (bar_w + gap)) - (total_w/2) + (bar_w/2)
    
    # choose colors automatically
    colors = {
        'baseline': '#ffb6a3',  # Black for baseline
        'reuse': '#7593af',  # Blue
        'sa': '#194a7a',  # Green
        'lookahead': '#7593af',  # Orange
        'best': '#4b86b4',  # Red
    }

    
    # Plot bars for non-baseline methods
    bar_idx = 0
    for i, m in enumerate(methods):
        if m.lower() == 'baseline':
            continue
        xs = index + offsets[bar_idx]
        # Custom label formatting
        if m.lower() == 'sa':
            label = 'SA'
        else:
            label = m.capitalize()
            
        ax.bar(
            xs,
            scaled[m],
            width=bar_w,
            label=label,
            color=colors[m.lower()],
            zorder=2
        )
        for xi, real_val in zip(xs, norm[m]):
            # Position text slightly below the top of bar (0.02 offset)
            y_pos = piecewise_scale(real_val, lower_dst=LOWER_DST) - 0.02
            ax.text(xi, y_pos,
                   f"{real_val:.2f}", 
                   ha='center', 
                   va='top',  # Changed from 'bottom' to 'top'
                   fontsize=9,
                   fontweight='bold',
                   color='white')
        bar_idx += 1

    # Plot baseline as horizontal line
    ax.axhline(y=piecewise_scale(1.0, lower_dst=LOWER_DST), 
               color='black', 
               linestyle='--', 
               label='Baseline',
               zorder=3,  # line above the bars
               linewidth=1.0)
        
    # axes labels & ticks
    ax.set_xlabel('Attention Sparsity (%)', fontsize=14, labelpad=30)
    ax.set_ylabel('Normalized tokens/sec', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels([f"{s}%" for s in sparsities])
    
    # custom y‐ticks back‐mapped to “real” normalized values
    y_max = max(max(norm[m]) for m in methods) * 1.1
    real_ticks = np.concatenate([
        np.linspace(0, 1, 6, endpoint=True),
        np.arange(1.1, np.ceil(y_max*10)/10 + 0.1, 0.1)
    ])

    # real_ticks = np.arange(1.0, np.ceil(y_max*10)/10 + 0.1, 0.1) 

    plot_ticks = [piecewise_scale(t,
                                  lower_src=(0.0, 1.0),
                                  lower_dst=LOWER_DST)
                  for t in real_ticks]
    ax.set_yticks(plot_ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in real_ticks])
    ax.set_ylim(0, piecewise_scale(y_max,
                                   lower_src=(0.0, 1.0),
                                   lower_dst=LOWER_DST))
    # ax.set_ylim(piecewise_scale(0, lower_dst=LOWER_DST),  # Start y-axis slightly below 1.0
    #             piecewise_scale(y_max, lower_src=(0.0, 1.0), lower_dst=LOWER_DST))
    
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    ax.legend(loc='upper left', fontsize=12, frameon=True)
    
    plt.tight_layout()
    plt.savefig("infer.png", dpi=300, bbox_inches="tight")
    plt.show()
