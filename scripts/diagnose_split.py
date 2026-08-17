#!/usr/bin/env python3
"""Diagnostic: does RUL become learnable with a within-condition train/test split?"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
features = pd.read_csv("taba_femto_outputs/features_reference.csv")
feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]

CONDITIONS = {
    1: ("Bearing1_1", "Bearing1_2"),
    2: ("Bearing2_1", "Bearing2_2"),
    3: ("Bearing3_1", "Bearing3_2"),
}

MODELS = {
    "Ridge": lambda: Ridge(alpha=1.0, random_state=RANDOM_STATE),
    "HistGradientBoosting": lambda: HistGradientBoostingRegressor(random_state=RANDOM_STATE),
    "RandomForest": lambda: RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1),
}

print("=== Within-condition split (train on one bearing, test on the other, same condition) ===")
for cond, (train_b, test_b) in CONDITIONS.items():
    train_df = features[features["bearing"] == train_b]
    test_df = features[features["bearing"] == test_b]
    scaler = StandardScaler().fit(train_df[feature_columns])
    X_train = scaler.transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    y_train = train_df["rul_min"].to_numpy()
    y_test = test_df["rul_min"].to_numpy()

    print(f"\nCondition {cond}: train={train_b} ({len(train_df)}), test={test_b} ({len(test_df)})")
    for name, make in MODELS.items():
        model = make()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"  {name:22s} MAE={mae:8.2f} min  R2={r2:6.3f}")
    # also reverse direction
    model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    model.fit(scaler.transform(test_df[feature_columns]), y_test)
    pred_rev = model.predict(scaler.transform(train_df[feature_columns]))
    print(f"  (reverse) RandomForest MAE={mean_absolute_error(y_train, pred_rev):8.2f} min  "
          f"R2={r2_score(y_train, pred_rev):6.3f}")

print("\n=== Sanity check: same-bearing train/test (random row split, should be easy) ===")
from sklearn.model_selection import train_test_split
b = features[features["bearing"] == "Bearing1_1"]
Xtr, Xte, ytr, yte = train_test_split(b[feature_columns], b["rul_min"], test_size=0.25, random_state=RANDOM_STATE)
scaler = StandardScaler().fit(Xtr)
model = RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
model.fit(scaler.transform(Xtr), ytr)
pred = model.predict(scaler.transform(Xte))
print(f"Bearing1_1 self-split RandomForest MAE={mean_absolute_error(yte, pred):.2f} min  R2={r2_score(yte, pred):.3f}")
