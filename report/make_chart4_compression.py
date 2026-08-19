#!/usr/bin/env python3
"""Figure 3: model-compression comparison (size and energy), same visual
style as chart2_model_size.png, for the report's new compression section."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

labels = ["RandomForest\n(reference,\nuncompressed)", "HistGB\n(top-10-features,\nprevious best)",
          "RandomForest\n(compressed +\ntop-10-features)"]
size_kb = [37656.8, 329.1, 285.2]
energy_kwh = [9.11e-6, 1.31e-6, 6.50e-7]
colors = ["#e34948", "#eb6834", "#1baf7a"]

fig, axes = plt.subplots(1, 2, figsize=(9, 4.2))

bars0 = axes[0].bar(labels, size_kb, color=colors)
axes[0].set_yscale("log")
axes[0].set_ylabel("Model size (KB, log scale)")
axes[0].set_title("Model size")
for b, v in zip(bars0, size_kb):
    axes[0].annotate(f"{v:,.0f} KB", (b.get_x() + b.get_width()/2, v), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=9)

bars1 = axes[1].bar(labels, energy_kwh, color=colors)
axes[1].set_yscale("log")
axes[1].set_ylabel("Training energy (kWh, log scale)")
axes[1].set_title("Training energy")
for b, v in zip(bars1, energy_kwh):
    axes[1].annotate(f"{v:.2e}", (b.get_x() + b.get_width()/2, v), xytext=(0, 5),
                      textcoords="offset points", ha="center", fontsize=9)

for ax in axes:
    ax.tick_params(axis="x", labelsize=8.5)

fig.suptitle("Effect of pruning + quantization on RandomForest, vs. the prior best configuration", fontsize=11.5, y=1.03)
plt.tight_layout()
plt.savefig("/home/user/EmergingTechnologiesAI/report/chart4_compression.png", dpi=200, bbox_inches="tight")
print("done")
