import matplotlib.pyplot as plt
import numpy as np

# Data from your table
data = {
    'decode_tokens': [2048, 4096, 8192, 16384],
    'Upper Bound': [12.8578, 26.4739, 55.9751, 124.0534],
    'Baseline': [15.2122, 30.9018, 65.9189, 191.6213],
    'Self-derived': [12.8763, 27.2510, 69.9974, 214.1309]
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
        data['decode_tokens'],
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
ax.set_xlabel('Decode Token Number', fontsize=12)
ax.set_ylabel('Time Taken (seconds)', fontsize=12)
ax.set_xticks(data['decode_tokens'])
ax.set_xticklabels([str(x) for x in data['decode_tokens']])
ax.set_ylim(0, 220)  # Adjust upper limit if needed
ax.grid(True, linestyle='--', alpha=0.7)
ax.legend(loc='upper left', fontsize=10)

# Add data labels
for method in methods:
    for x, y in zip(data['decode_tokens'], data[method]):
        ax.text(
            x, y + 5, f'{y:.1f}',
            ha='center', va='bottom',
            fontsize=8, color=methods[method]['color']
        )

plt.tight_layout()
plt.savefig('inference_time_comparison.png', dpi=300)
plt.show()