#!/usr/bin/env python3
"""Combined Green AI experiment: model-compression (pruning + quantization,
per Week 2/3 course material) x data-centric strategy (top-10-features),
tested on the 3 models where compression is meaningful (RandomForest,
HistGradientBoosting, MLP - Ridge is already <1KB and excluded).

Compression definitions (structured pruning preferred over unstructured,
per course guidance):
  RandomForest:          300 trees, unlimited depth -> 50 trees, max_depth=10
                          + quantization (float64 -> float32 tree payload,
                            measured via manual quantized tree-walk since
                            sklearn's Tree Cython object can't be quantized
                            in place)
  HistGradientBoosting:  max_iter=100, max_leaf_nodes=31 (defaults)
                          -> max_iter=30, max_leaf_nodes=15 (pruned).
                          No quantization applied: HGB's internal predictor
                          uses a structured numpy dtype not safely
                          quantizable without relying on private sklearn
                          internals - documented limitation, not attempted.
  MLP:                   hidden_layer_sizes=(64,32) -> (16,8) (pruned)
                          + coefs_/intercepts_ cast to float32 in place
                          (quantization; MLP's plain ndarray attributes make
                          this a genuine, safe, real quantization unlike RF).

Each is tested under 'reference' (24 features) and 'top-10-features' (10
features, RandomForest-importance-ranked) data strategies, within-condition
split (train one bearing, test its sibling), across all 3 conditions.
"""
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.neural_network import MLPRegressor
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


def top_k_features(X_train, y_train, columns, k=10):
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [columns[i] for i in order[:k]]


def rf_quantized_predict_and_payload(model, X):
    payload_bytes = 0
    preds = np.zeros((len(model.estimators_), X.shape[0]))
    for i, tree in enumerate(model.estimators_):
        t = tree.tree_
        threshold_q = t.threshold.astype(np.float32)
        value_q = t.value.astype(np.float32)
        payload_bytes += threshold_q.nbytes + value_q.nbytes
        node = np.zeros(X.shape[0], dtype=int)
        for _ in range(t.max_depth + 1):
            is_leaf = t.children_left[node] == t.children_right[node]
            feat = t.feature[node]
            go_left = (X[np.arange(X.shape[0]), np.clip(feat, 0, X.shape[1] - 1)] <= threshold_q[node])
            next_node = np.where(go_left, t.children_left[node], t.children_right[node])
            node = np.where(is_leaf, node, next_node)
        preds[i] = value_q[node, 0, 0]
    return preds.mean(axis=0), payload_bytes / 1024


MODEL_CONFIGS = {
    "RandomForest": {
        "uncompressed": dict(n_estimators=300, max_depth=None),
        "compressed": dict(n_estimators=50, max_depth=10),
    },
    "HistGradientBoosting": {
        "uncompressed": dict(max_iter=100, max_leaf_nodes=31),
        "compressed": dict(max_iter=30, max_leaf_nodes=15),
    },
    "MLP": {
        "uncompressed": dict(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True),
        "compressed": dict(hidden_layer_sizes=(16, 8), max_iter=2000, early_stopping=True),
    },
}
STRATEGIES = ["reference", "top-10-features"]

results = []
for cond, (train_b, test_b) in CONDITIONS.items():
    train_df = features[features["bearing"] == train_b].sort_values("obs_index").reset_index(drop=True)
    test_df = features[features["bearing"] == test_b].sort_values("obs_index").reset_index(drop=True)

    for strategy in STRATEGIES:
        if strategy == "reference":
            cols = feature_columns
        else:
            scaler_tmp = StandardScaler().fit(train_df[feature_columns])
            cols = top_k_features(scaler_tmp.transform(train_df[feature_columns]),
                                   train_df["rul_frac"].to_numpy(), feature_columns, k=10)

        scaler = StandardScaler().fit(train_df[cols])
        X_train = scaler.transform(train_df[cols])
        X_test = scaler.transform(test_df[cols])
        y_train = train_df["rul_frac"].to_numpy()
        y_test = test_df["rul_frac"].to_numpy()

        for model_name, variants in MODEL_CONFIGS.items():
            for variant_name, params in variants.items():
                if model_name == "RandomForest":
                    model = RandomForestRegressor(random_state=RANDOM_STATE, n_jobs=-1, **params)
                elif model_name == "HistGradientBoosting":
                    model = HistGradientBoostingRegressor(random_state=RANDOM_STATE, **params)
                else:
                    model = MLPRegressor(random_state=RANDOM_STATE, **params)

                _, train_time, train_energy_kwh, train_co2_g = measure(lambda: model.fit(X_train, y_train))

                if model_name == "RandomForest" and variant_name == "compressed":
                    (y_pred, size_kb), infer_time, infer_energy_kwh, infer_co2_g = measure(
                        lambda: rf_quantized_predict_and_payload(model, X_test))
                    quant_note = "pruned+quantized(payload)"
                else:
                    if model_name == "MLP" and variant_name == "compressed":
                        model.coefs_ = [c.astype(np.float32) for c in model.coefs_]
                        model.intercepts_ = [b.astype(np.float32) for b in model.intercepts_]
                        quant_note = "pruned+quantized(real)"
                    elif model_name == "HistGradientBoosting" and variant_name == "compressed":
                        quant_note = "pruned only(no quant)"
                    else:
                        quant_note = "baseline"
                    y_pred, infer_time, infer_energy_kwh, infer_co2_g = measure(lambda: model.predict(X_test))
                    size_kb = len(pickle.dumps(model)) / 1024

                mae = mean_absolute_error(y_test, y_pred)
                r2 = r2_score(y_test, y_pred)
                truly_critical = y_test <= CRITICAL_FRAC
                recall = (float(((y_pred <= CRITICAL_FRAC) & truly_critical).sum() / truly_critical.sum())
                          if truly_critical.sum() > 0 else float("nan"))
                lt = lead_time_minutes(test_df, y_pred)

                results.append({
                    "condition": cond, "strategy": strategy, "model": model_name, "variant": variant_name,
                    "n_features": len(cols), "quant_method": quant_note,
                    "mae_frac": mae, "r2": r2, "critical_recall": recall, "lead_time_min": lt,
                    "train_time_s": train_time, "train_energy_kwh": train_energy_kwh,
                    "co2_g": train_co2_g + infer_co2_g, "size_kb": size_kb,
                })
                print(f"cond{cond} {strategy:16s} {model_name:22s} [{variant_name:12s}] "
                      f"MAE={mae:.3f} R2={r2:6.3f} recall={recall:.2f} lead={lt:5.1f}min "
                      f"size={size_kb:9.2f}KB energy={train_energy_kwh:.2e}kWh")

results_df = pd.DataFrame(results)
results_df.to_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/combined_green_ai_results.csv", index=False)

print("\n=== Aggregated (mean across 3 conditions) ===")
agg = results_df.groupby(["strategy", "model", "variant"]).agg(
    mae_frac=("mae_frac", "mean"), r2=("r2", "mean"), critical_recall=("critical_recall", "mean"),
    lead_time_min=("lead_time_min", "mean"), train_energy_kwh=("train_energy_kwh", "mean"),
    co2_g=("co2_g", "mean"), size_kb=("size_kb", "mean"), n_features=("n_features", "first"),
).reset_index()
print(agg.to_string())
agg.to_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/combined_green_ai_aggregated.csv", index=False)
