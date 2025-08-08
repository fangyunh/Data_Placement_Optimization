
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
    plt.figure(figsize=(10, 4), dpi=300)
    ax = plt.gca()

    # Define plotting order and filter out baseline and best
    plot_order = [m for m in tar_methods if m.lower() not in ['best']]
    n_methods = len(plot_order)  # Only count methods that will be bars
    
    # bar geometry
    # n_methods = len(methods) - 1
    bar_w = 0.1
    gap = 0.01
    group_gap = 0.2
    total_w = n_methods * bar_w + (n_methods - 1) * gap
    # x positions for each group
    index = np.arange(len(randoms)) * (total_w + group_gap)
    # offsets to center each bar in a group
    offsets = (np.arange(n_methods) * (bar_w + gap)) - (total_w/2) + (bar_w/2)
    
    # choose colors automatically
    colors = {
        'baseline': '#5a6d8c',  # Black for baseline
        'reuse': '#baccd9',  # Blue
        'page': '#5697c3',  # Orange
        'sa': '#11659a',  # Green
        'best': '#5a6d8c',  # Red
    }

    
    # Plot bars for non-baseline methods
    bar_idx = 0
    for m in plot_order:
        if m.lower() == ['best']:
            continue
        xs = index + offsets[bar_idx]
        # Custom label formatting
        label = display_names[m.lower()]
            
        ax.bar(
            xs,
            scaled[m],
            width=bar_w,
            label=label,
            color=colors[m.lower()],
            zorder=2
        )
        # for xi, real_val in zip(xs, norm[m]):
        #     # Position text slightly below the top of bar (0.02 offset)
        #     y_pos = piecewise_scale(real_val, lower_dst=LOWER_DST) - 0.02
        #     ax.text(xi, y_pos,
        #            f"{real_val:.2f}", 
        #            ha='center', 
        #            va='top',  # Changed from 'bottom' to 'top'
        #            fontsize=8,
        #            fontweight='bold',
        #            color='white')
        bar_idx += 1

    # Plot baseline as horizontal line
    # Update baseline line to show its relative performance to best
    # ax.axhline(y=piecewise_scale(norm['baseline'][0], lower_dst=LOWER_DST), 
    #            color='black', 
    #            linestyle='--', 
    #            label=display_names['baseline'],
    #            zorder=3,
    #            linewidth=1.0)

    # Plot best case line connecting normalized points
    # best_ys = [piecewise_scale(y, lower_dst=LOWER_DST) for y in norm['best']]
    # ax.plot(index, best_ys,
    #         color='#126d82',
    #         linestyle='-.',
    #         label=display_names['best'],
    #         zorder=3,
    #         linewidth=1.0)
    
    # # Add scatter points on top
    # ax.scatter(index, best_ys,
    #           color='#126d82',
    #           s=50,  # point size
    #           zorder=4,  # ensure points are above line
    #           marker='o')  # circular markers

        
    # axes labels & ticks
    ax.set_xlabel('Randomness', fontsize=14)
    ax.set_ylabel('Normalized tokens/sec', fontsize=14)
    ax.set_xticks(index)
    ax.set_xticklabels(tar_random)
    
    # custom y‐ticks back‐mapped to “real” normalized values
    y_max = max(max(norm[m]) for m in methods) * 1.3
    y_ticks = np.arange(0, 1.1, 0.1)  # Ticks from 0 to 1 in 0.1 steps
    ax.set_yticks(y_ticks)
    ax.set_yticklabels([f"{t:.1f}" for t in y_ticks])
    ax.set_ylim(0, 1.0)
    # ax.set_ylim(piecewise_scale(0, lower_dst=LOWER_DST),  # Start y-axis slightly below 1.0
    #             piecewise_scale(y_max, lower_src=(0.0, 1.0), lower_dst=LOWER_DST))
    
    
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    # ax.legend(loc='upper right', fontsize=12, frameon=True)
    ax.legend(
        loc='upper center',  # Position at top center
        bbox_to_anchor=(0.5, 1.15),  # Adjust vertical position above plot
        ncol=5,  # Show all items in one row
        fontsize=9,
        frameon=True,
        borderaxespad=0.
    )
    
    plt.tight_layout()
    plt.savefig("random.png", dpi=300, bbox_inches="tight")
    plt.show()
