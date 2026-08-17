#!/usr/bin/env python3
"""Static PNG charts for the Word report (RandomForest used as the consistent
example model in the diagnostic chart, to isolate the effect of each fix)."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 11,
    "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": "#0b0b0b",
    "text.color": "#0b0b0b",
    "xtick.color": "#52514e",
    "ytick.color": "#52514e",
    "axes.grid": True,
    "grid.color": "#e1e0d9",
    "grid.linewidth": 0.7,
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

SERIES = {"half-rate": "#2a78d6", "reference": "#eb6834", "top-10-features": "#1baf7a"}
MARKERS = {"Ridge": "o", "HistGradientBoosting": "s", "RandomForest": "^", "MLP": "D"}

df = pd.read_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/final_results_aggregated.csv")

# ---------- Chart 1: Pareto Energy x Critical Recall / Lead Time ----------
fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))

for ax, ykey, ylabel, yfmt in [
    (axes[0], "lead_time_min", "Lead time (min)", None),
    (axes[1], "critical_recall", "Critical recall", "pct"),
]:
    for _, row in df.iterrows():
        ax.scatter(row["train_energy_kwh"], row[ykey],
                   color=SERIES[row["strategy"]], marker=MARKERS[row["model"]],
                   s=90, edgecolor="white", linewidth=1.2, zorder=3)
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
fig.legend(handles=strategy_handles, title="Strategy (color)", loc="upper center",
           bbox_to_anchor=(0.28, 0.02), ncol=1, frameon=False, fontsize=9, title_fontsize=9)
fig.legend(handles=model_handles, title="Model (shape)", loc="upper center",
           bbox_to_anchor=(0.72, 0.02), ncol=1, frameon=False, fontsize=9, title_fontsize=9)
fig.suptitle("Energy vs. predictive-maintenance viability (averaged across 3 conditions)", fontsize=12, y=1.02)
plt.tight_layout(rect=[0, 0.14, 1, 1])
plt.savefig("/home/user/EmergingTechnologiesAI/report/chart1_pareto.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Chart 2: Model size comparison (reference strategy) ----------
size_df = df[df["strategy"] == "reference"].sort_values("model_size_kb")
fig, ax = plt.subplots(figsize=(7, 4))
colors = ["#2a78d6", "#eb6834", "#1baf7a", "#e34948"]
bars = ax.bar(size_df["model"], size_df["model_size_kb"], color=colors[:len(size_df)])
ax.set_yscale("log")
ax.set_ylabel("Serialized model size (KB, log scale)")
ax.set_title("Model size by algorithm (reference configuration)")
for b, v in zip(bars, size_df["model_size_kb"]):
    label = f"{v:,.0f} KB" if v >= 1 else f"{v:.2f} KB"
    ax.annotate(label, (b.get_x() + b.get_width() / 2, v), xytext=(0, 5),
                textcoords="offset points", ha="center", fontsize=9)
plt.xticks(rotation=10)
plt.tight_layout()
plt.savefig("/home/user/EmergingTechnologiesAI/report/chart2_model_size.png", dpi=200, bbox_inches="tight")
plt.close()

# ---------- Chart 3: Diagnostic journey (RandomForest, held constant) ----------
stages = [
    "Same-bearing\nholdout\n(sanity check)",
    "Cross-condition split\nabsolute RUL\n(original attempt)",
    "Within-condition split\nabsolute RUL\n(condition 2)",
    "Within-condition split\nnormalized RUL\n(condition 2)",
]
values = [0.993, -0.303, 0.307, 0.615]
colors = ["#1baf7a" if v >= 0 else "#e34948" for v in values]

fig, ax = plt.subplots(figsize=(8, 4.5))
bars = ax.bar(stages, values, color=colors, width=0.6)
ax.axhline(0, color="#898781", linewidth=1)
ax.set_ylabel("R² (RandomForest)")
ax.set_title("R² across the methodology fixes (RandomForest held constant)")
for b, v in zip(bars, values):
    ax.annotate(f"{v:.2f}", (b.get_x() + b.get_width() / 2, v),
                xytext=(0, 6 if v >= 0 else -14), textcoords="offset points",
                ha="center", fontsize=10, fontweight="bold")
plt.xticks(fontsize=9)
plt.tight_layout()
plt.savefig("/home/user/EmergingTechnologiesAI/report/chart3_diagnostic_journey.png", dpi=200, bbox_inches="tight")
plt.close()

print("done")
