import matplotlib.pyplot as plt
import numpy as np

# Data from your table (in seconds)
# 8GB
# data = {
#     'HBM sizes': [0, 20, 40, 60],
#     'Baseline': [320.4148, 317.6644, 301.5573, 248.5209],
#     'Inclusive': [314.8761, 255.1913, 206.6139, 183.6477]
# }

data = {
    'HBM sizes': [0, 20, 40, 60],
    'Baseline': [395.5111, 382.9213, 320.3195, 248.8375],
    'Inclusive': [393.4218, 322.832, 256.0797, 198.9209]
}

# Convert time to tokens per second
decode_token = 1024 * 24  # Total number of tokens
real_decode = [decode_token, decode_token * 0.8, decode_token * 0.6, decode_token * 0.4]
# Calculate tokens per second using real decoded tokens
tokens_per_second = {
    'Baseline': [real_decode[i]/t for i, t in enumerate(data['Baseline'])],
    'Inclusive': [real_decode[i]/t for i, t in enumerate(data['Inclusive'])]
}

# Normalize to baseline at 0% sparsity
# baseline_0 = tokens_per_second['Baseline'][0]
normalized_throughput = {
    'Baseline': [1.0] * len(data['HBM sizes']),  # Baseline is always 1.0
    'Inclusive': [inc/base for inc, base in zip(tokens_per_second['Inclusive'], 
                                              tokens_per_second['Baseline'])]
}



# Create figure and axis with higher resolution
plt.figure(figsize=(10, 6), dpi=300)
ax = plt.gca()

# Plot each method with improved styling
methods = {
    'Baseline': {
        'color': '#9a82cb', 
        'marker': '^',
        'label': 'Baseline'
    },
    'Inclusive': {
        'color': '#ff8168', 
        'marker': 'o',
        'label': 'LookAheadOne'
    }
}

for method, style in methods.items():
    ax.plot(
        data['HBM sizes'],
        normalized_throughput[method],
        label=style['label'],
        color=style['color'],
        marker=style['marker'],
        markersize=8,
        linewidth=2,
        linestyle='--'
    )

# Customize plot with improved labels
ax.set_xlabel('KV Cache Sparse Ratio (%)', fontsize=18)
ax.set_ylabel('Relative token per second', fontsize=18)
ax.set_xticks(data['HBM sizes'])
ax.set_xticklabels([f'{x}%' for x in data['HBM sizes']])

# Add grid and legend with improved visibility
ax.grid(True, linestyle='--', alpha=0.7)
legend = ax.legend(
    loc='upper left',
    fontsize=18,
    framealpha=0.9,  # More visible legend background
    edgecolor='black',  # Add border to legend
    fancybox=True,  # Rounded corners
    borderpad=1  # Padding inside legend border
)

# Add data labels for tokens per second
for method in methods:
    for x, y in zip(data['HBM sizes'], normalized_throughput[method]):
        ax.text(
            x, y + (max(normalized_throughput[method])*0.03),  # Adjusted offset
            f'{y:.1f}',
            ha='center', 
            va='bottom',
            fontsize=10, 
            color=methods[method]['color'],
            fontweight='bold'
        )
# After calculating normalized_throughput but before plotting
min_value = min(min(normalized_throughput['Baseline']), min(normalized_throughput['Inclusive']))
max_value = max(max(normalized_throughput['Baseline']), 
               max(normalized_throughput['Inclusive']))
y_max = max_value * 1.1  # Add 20% padding above maximum value

# After plotting but before plt.show()
ax.set_ylim(min_value * 0.95, y_max)

# Adjust layout and save
plt.tight_layout()
plt.savefig('inference_throughput_comparison.png', dpi=300, bbox_inches='tight')
plt.show()