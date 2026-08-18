#!/usr/bin/env python3
"""Green AI strategy #3 (exploratory): a single pooled model across all 3
operating conditions, with condition (speed_rpm, load_N) as explicit input
features, instead of training 3 separate condition-specific models.

Train: one bearing per condition (Bearing1_1, Bearing2_1, Bearing3_1), pooled.
Test: the sibling bearing of each condition (Bearing1_2, Bearing2_2, Bearing3_2),
evaluated per-condition so results are directly comparable to Table 2 in the report.
"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
CRITICAL_FRAC = 0.20

features = pd.read_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/features_reference.csv")
feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]
features["rul_frac"] = features["rul_min"] / features.groupby("bearing")["rul_min"].transform("max")

CONDITION_PHYSICS = {
    "Bearing1_1": (1800, 4000), "Bearing1_2": (1800, 4000),
    "Bearing2_1": (1650, 4200), "Bearing2_2": (1650, 4200),
    "Bearing3_1": (1500, 5000), "Bearing3_2": (1500, 5000),
}
features["speed_rpm"] = features["bearing"].map(lambda b: CONDITION_PHYSICS[b][0])
features["load_n"] = features["bearing"].map(lambda b: CONDITION_PHYSICS[b][1])

TRAIN_BEARINGS = ["Bearing1_1", "Bearing2_1", "Bearing3_1"]
TEST_BEARINGS = {"Bearing1_1": "Bearing1_2", "Bearing2_1": "Bearing2_2", "Bearing3_1": "Bearing3_2"}
COND_LABEL = {"Bearing1_1": 1, "Bearing2_1": 2, "Bearing3_1": 3}

cols_with_condition = feature_columns + ["speed_rpm", "load_n"]

train_df = features[features["bearing"].isin(TRAIN_BEARINGS)].reset_index(drop=True)

MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
}

def lead_time_minutes(test_df_sorted, y_pred_frac):
    truly_critical = test_df_sorted["rul_frac"].to_numpy() <= CRITICAL_FRAC
    if not truly_critical.any():
        return np.nan
    pred_critical = y_pred_frac <= CRITICAL_FRAC
    correct_alarm = truly_critical & pred_critical
    if not correct_alarm.any():
        return 0.0
    first_idx = np.argmax(correct_alarm)
    return float(test_df_sorted["rul_min"].to_numpy()[first_idx])

print("=== Pooled model, condition (speed_rpm, load_n) as explicit features ===")
print(f"Train bearings (pooled): {TRAIN_BEARINGS}\n")

results = []
for model_name, make in MODELS.items():
    scaler = StandardScaler().fit(train_df[cols_with_condition])
    X_train = scaler.transform(train_df[cols_with_condition])
    y_train = train_df["rul_frac"].to_numpy()
    model = make()
    model.fit(X_train, y_train)

    print(f"--- {model_name} (single pooled model) ---")
    for train_b, test_b in TEST_BEARINGS.items():
        cond = COND_LABEL[train_b]
        test_df = features[features["bearing"] == test_b].sort_values("obs_index").reset_index(drop=True)
        X_test = scaler.transform(test_df[cols_with_condition])
        y_test = test_df["rul_frac"].to_numpy()
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        truly_critical = y_test <= CRITICAL_FRAC
        if truly_critical.sum() > 0:
            pred_flagged = y_pred <= CRITICAL_FRAC
            recall = float((pred_flagged & truly_critical).sum() / truly_critical.sum())
        else:
            recall = float("nan")
        lt = lead_time_minutes(test_df, y_pred)

        results.append({"model": model_name, "condition": cond, "test_bearing": test_b,
                         "mae_frac": mae, "r2": r2, "critical_recall": recall, "lead_time_min": lt})
        print(f"  condition {cond} ({test_b}): MAE={mae:.3f} R2={r2:6.3f} "
              f"recall={recall:.2f} lead_time={lt:.1f}min")
    print()

results_df = pd.DataFrame(results)
results_df.to_csv("/home/user/EmergingTechnologiesAI/taba_femto_outputs/condition_aware_results.csv", index=False)
print("Saved to condition_aware_results.csv")
