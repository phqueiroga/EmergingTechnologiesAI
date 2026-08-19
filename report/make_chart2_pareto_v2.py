#!/usr/bin/env python3
"""Regenerate Figure 2 (Pareto), adding the compressed model-compression
variants as hollow-ring-marked points overlaid on the original chart."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd

plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 11,
    "axes.edgecolor": "#c3c2b7", "axes.labelcolor": "#0b0b0b", "text.color": "#0b0b0b",
    "xtick.color": "#52514e", "ytick.color": "#52514e",
    "axes.grid": True, "grid.color": "#e1e0d9", "grid.linewidth": 0.7,
    "figure.facecolor": "white", "axes.facecolor": "white",
})

SERIES = {"half-rate": "#2a78d6", "reference": "#eb6834", "top-10-features": "#1baf7a"}
MARKERS = {"Ridge": "o", "HistGradientBoosting": "s", "RandomForest": "^", "MLP": "D"}

df = pd.read_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/final_results_aggregated.csv")
comp = pd.read_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/combined_green_ai_aggregated.csv")
comp = comp[comp["variant"] == "compressed"].copy()
comp = comp.rename(columns={"critical_recall": "recall", "size_kb": "model_size_kb"})

fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

for ax, ykey, ylabel, yfmt in [
    (axes[0], "lead_time_min", "Lead time (min)", None),
    (axes[1], "critical_recall", "Critical recall", "pct"),
]:
    for _, row in df.iterrows():
        ax.scatter(row["train_energy_kwh"], row[ykey],
                   color=SERIES[row["strategy"]], marker=MARKERS[row["model"]],
                   s=90, edgecolor="white", linewidth=1.2, zorder=3)

    comp_ykey = "lead_time_min" if ykey == "lead_time_min" else "recall"
    for _, row in comp.iterrows():
        ax.scatter(row["train_energy_kwh"], row[comp_ykey],
                   facecolor=SERIES[row["strategy"]], marker=MARKERS[row["model"]],
                   s=90, edgecolor="black", linewidth=1.4, zorder=4)
        ax.scatter(row["train_energy_kwh"], row[comp_ykey],
                   facecolor="none", marker="o", s=220, edgecolor="black",
                   linewidth=1.3, linestyle=(0, (2, 2)), zorder=5)

    ax.set_xscale("log")
    ax.set_xlabel("Training energy (kWh, log scale)")
    ax.set_ylabel(ylabel)
    if yfmt == "pct":
        ax.yaxis.set_major_formatter(mticker.PercentFormatter(xmax=1))
    ax.set_ylim(bottom=0)
    ax.grid(True, which="major", axis="both")

strategy_handles = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=c, markersize=9, label=s)
                     for s, c in SERIES.items()]
model_handles = [plt.Line2D([0], [0], marker=m, color="#52514e", linestyle="", markersize=8, label=k)
                  for k, m in MARKERS.items()]
compressed_handle = [plt.Line2D([0], [0], marker="o", color="w", markerfacecolor="#c3c2b7",
                                 markeredgecolor="black", markeredgewidth=1.3, markersize=11,
                                 label="Compressed (pruned + quantized)")]

fig.legend(handles=strategy_handles, title="Strategy (color)", loc="upper center",
           bbox_to_anchor=(0.20, 0.02), ncol=1, frameon=False, fontsize=9, title_fontsize=9)
fig.legend(handles=model_handles, title="Model (shape)", loc="upper center",
           bbox_to_anchor=(0.55, 0.02), ncol=1, frameon=False, fontsize=9, title_fontsize=9)
fig.legend(handles=compressed_handle, title="Compression (ring)", loc="upper center",
           bbox_to_anchor=(0.90, 0.02), ncol=1, frameon=False, fontsize=9, title_fontsize=9)

fig.suptitle("Energy vs. predictive-maintenance viability (averaged across 3 conditions)", fontsize=12, y=1.02)
plt.tight_layout(rect=[0, 0.16, 1, 1])
plt.savefig("/home/user/EmergingTechnologiesAI/report/chart1_pareto_v2.png", dpi=200, bbox_inches="tight")
print("done")
