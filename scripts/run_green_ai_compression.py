#!/usr/bin/env python3
"""Green AI model-compression strategies (as taught in Week 2 / Peykani et al.
review): pruning and low-precision (quantization) applied to RandomForest,
the model already flagged as disproportionately expensive in the main study.

Strategy A - Pruning: fewer/shallower trees (300 trees, unlimited depth ->
50 trees, max_depth=10). Structural pruning of the ensemble.

Strategy B - Quantization (low-precision computing): cast the trained
model's internal float64 arrays to float32 before serialization/inference,
halving numeric precision as recommended in the reading (FP16/INT8 not
supported by sklearn natively; float32 is the feasible equivalent here).

Both are compared against the original RandomForest (reference config,
within-condition split, same as the main study) for size, energy, MAE, R2,
critical recall and lead time.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score
from codecarbon import OfflineEmissionsTracker

RANDOM_STATE = 42
CRITICAL_FRAC = 0.20

features = pd.read_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/features_reference.csv")
feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]
features["rul_frac"] = features["rul_min"] / features.groupby("bearing")["rul_min"].transform("max")

CONDITIONS = {1: ("Bearing1_1", "Bearing1_2"), 2: ("Bearing2_1", "Bearing2_2"), 3: ("Bearing3_1", "Bearing3_2")}


def measure(fn):
    tracker = OfflineEmissionsTracker(log_level="error", save_to_file=False, country_iso_code="IRL")
    tracker.start()
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    tracker.stop()
    energy_kwh = tracker.final_emissions_data.energy_consumed
    co2_g = tracker.final_emissions_data.emissions * 1000
    return result, elapsed, energy_kwh, co2_g


def lead_time_minutes(test_df_sorted, y_pred_frac):
    truly_critical = test_df_sorted["rul_frac"].to_numpy() <= CRITICAL_FRAC
    if not truly_critical.any():
        return np.nan
    pred_critical = y_pred_frac <= CRITICAL_FRAC
    correct_alarm = truly_critical & pred_critical
    if not correct_alarm.any():
        return 0.0
    return float(test_df_sorted["rul_min"].to_numpy()[np.argmax(correct_alarm)])


def quantize_predict(model, X, dtype):
    """Predict using the model's tree values/thresholds truncated to `dtype`
    precision (this is what actually changes under quantization: predictions
    computed from lower-precision numbers), and return that precision's real
    payload size in KB (sum of threshold+value arrays across all trees, the
    part of the model quantization actually shrinks)."""
    payload_bytes = 0
    preds = np.zeros((len(model.estimators_), X.shape[0]))
    for i, tree in enumerate(model.estimators_):
        t = tree.tree_
        threshold_q = t.threshold.astype(dtype)
        value_q = t.value.astype(dtype)
        payload_bytes += threshold_q.nbytes + value_q.nbytes
        # walk the tree manually with quantized thresholds/values to get a
        # genuinely precision-affected prediction, not just a relabeled dtype
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(t.max_depth + 1):
            is_leaf = t.children_left[node] == t.children_right[node]
            feat = t.feature[node]
            go_left = (X[np.arange(X.shape[0]), np.clip(feat, 0, X.shape[1] - 1)] <= threshold_q[node])
            next_node = np.where(go_left, t.children_left[node], t.children_right[node])
            node = np.where(is_leaf, node, next_node)
        preds[i] = value_q[node, 0, 0]
    return preds.mean(axis=0), payload_bytes / 1024


VARIANTS = {
    "RF-reference (300 trees, full depth)": dict(n_estimators=300, max_depth=None),
    "RF-pruned (50 trees, max_depth=10)": dict(n_estimators=50, max_depth=10),
}

results = []
for cond, (train_b, test_b) in CONDITIONS.items():
    train_df = features[features["bearing"] == train_b].sort_values("obs_index").reset_index(drop=True)
    test_df = features[features["bearing"] == test_b].sort_values("obs_index").reset_index(drop=True)

    scaler = StandardScaler().fit(train_df[feature_columns])
    X_train = scaler.transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    y_train = train_df["rul_frac"].to_numpy()
    y_test = test_df["rul_frac"].to_numpy()

    for variant_name, params in VARIANTS.items():
        model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
        _, train_time, train_energy_kwh, train_co2_g = measure(lambda: model.fit(X_train, y_train))
        full_pickle_kb = len(pickle.dumps(model)) / 1024

        for precision_label, dtype in [("float64 (baseline)", np.float64), ("float32 (quantized)", np.float32)]:
            (y_pred, payload_kb), infer_time, infer_energy_kwh, infer_co2_g = measure(
                lambda: quantize_predict(model, X_test, dtype))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            truly_critical = y_test <= CRITICAL_FRAC
            recall = (float(((y_pred <= CRITICAL_FRAC) & truly_critical).sum() / truly_critical.sum())
                      if truly_critical.sum() > 0 else float("nan"))
            lt = lead_time_minutes(test_df, y_pred)

            results.append({
                "condition": cond, "variant": variant_name, "precision": precision_label,
                "mae_frac": mae, "r2": r2, "critical_recall": recall, "lead_time_min": lt,
                "train_time_s": train_time, "train_energy_kwh": train_energy_kwh,
                "co2_g": train_co2_g + infer_co2_g,
                "full_pickle_kb": full_pickle_kb, "numeric_payload_kb": payload_kb,
            })
            print(f"cond{cond} {variant_name:38s} [{precision_label:20s}] "
                  f"MAE={mae:.3f} R2={r2:6.3f} recall={recall:.2f} lead={lt:5.1f}min "
                  f"payload={payload_kb:8.1f}KB (full pickle={full_pickle_kb:9.1f}KB) energy={train_energy_kwh:.2e}kWh")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/green_ai_compression_results.csv", index=False)

print("\n=== Aggregated (mean across 3 conditions) ===")
agg = results_df.groupby(["variant", "precision"]).agg(
    mae_frac=("mae_frac", "mean"), r2=("r2", "mean"), critical_recall=("critical_recall", "mean"),
    lead_time_min=("lead_time_min", "mean"), train_energy_kwh=("train_energy_kwh", "mean"),
    co2_g=("co2_g", "mean"), full_pickle_kb=("full_pickle_kb", "mean"),
    numeric_payload_kb=("numeric_payload_kb", "mean"),
).reset_index()
print(agg.to_string())
agg.to_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/green_ai_compression_aggregated.csv", index=False)
