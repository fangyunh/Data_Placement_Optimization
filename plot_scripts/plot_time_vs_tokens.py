import matplotlib.pyplot as plt
import numpy as np

# Data from your table
data = {
    'HBM sizes': [4, 8, 12],
    'Upper Bound': [118.6087, 118.6048, 118.6048],
    'Baseline': [ 169.0090, 143.5742, 137.1847],
    'Self-derived': [ 190.6894, 148.3520, 124.4950]
}

# Create figure and axis
plt.figure(figsize=(10, 6))
ax = plt.gca()

# Plot each method
methods = {
    'Upper Bound': {'color': 'blue', 'marker': 's'},
    'Baseline': {'color': 'red', 'marker': '^'},
    'Self-derived': {'color': 'green', 'marker': 'o'}
}

for method, style in methods.items():
    ax.plot(
        data['HBM sizes'],
        data[method],
        label=method,
        color=style['color'],
        marker=style['marker'],
        markersize=8,
        linewidth=2,
        linestyle='--'
    )

# Customize plot
ax.set_title('Total Inference Time Comparison', fontsize=14, pad=20)
ax.set_xlabel('HBM sizes (GB)', fontsize=12)
ax.set_ylabel('Time Taken (seconds)', fontsize=12)
ax.set_xticks(data['HBM sizes'])
ax.set_xticklabels([str(x) for x in data['HBM sizes']])
ax.set_ylim(0, 220)  # Adjust upper limit if needed
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=10)

# Add data labels
for method in methods:
    for x, y in zip(data['HBM sizes'], data[method]):
        ax.text(
            x, y + 5, f'{y:.1f}',
            ha='center', va='bottom',
            fontsize=8, color=methods[method]['color']
        )

plt.tight_layout()
plt.savefig('inference_time_comparison.png', dpi=300)
plt.show()