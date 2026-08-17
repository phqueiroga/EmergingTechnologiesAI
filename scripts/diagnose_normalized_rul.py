#!/usr/bin/env python3
"""Diagnostic: does normalized RUL (fraction of life remaining) generalize better
across bearings than absolute rul_min?"""
import pandas as pd
import numpy as np
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

RANDOM_STATE = 42
features = pd.read_csv("taba_femto_outputs/features_reference.csv")
feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]

# Normalized RUL: fraction of remaining life, 0 (failure) to 1 (start of life).
features["rul_frac"] = features["rul_min"] / features.groupby("bearing")["rul_min"].transform("max")

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

print("=== Within-condition split, TARGET = rul_frac (normalized 0-1) ===")
for cond, (train_b, test_b) in CONDITIONS.items():
    train_df = features[features["bearing"] == train_b]
    test_df = features[features["bearing"] == test_b]
    scaler = StandardScaler().fit(train_df[feature_columns])
    X_train = scaler.transform(train_df[feature_columns])
    X_test = scaler.transform(test_df[feature_columns])
    y_train = train_df["rul_frac"].to_numpy()
    y_test = test_df["rul_frac"].to_numpy()

    print(f"\nCondition {cond}: train={train_b} ({len(train_df)}), test={test_b} ({len(test_df)})")
    for name, make in MODELS.items():
        model = make()
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        mae = mean_absolute_error(y_test, pred)
        r2 = r2_score(y_test, pred)
        print(f"  {name:22s} MAE={mae:6.3f} (frac)  R2={r2:6.3f}")

print("\n=== Cross-condition split (train cond2+3, test cond1), TARGET = rul_frac ===")
train_df = features[features["bearing"].isin(["Bearing2_1", "Bearing2_2", "Bearing3_1", "Bearing3_2"])]
test_df = features[features["bearing"].isin(["Bearing1_1", "Bearing1_2"])]
scaler = StandardScaler().fit(train_df[feature_columns])
X_train = scaler.transform(train_df[feature_columns])
X_test = scaler.transform(test_df[feature_columns])
y_train = train_df["rul_frac"].to_numpy()
y_test = test_df["rul_frac"].to_numpy()
for name, make in MODELS.items():
    model = make()
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, pred)
    r2 = r2_score(y_test, pred)
    print(f"  {name:22s} MAE={mae:6.3f} (frac)  R2={r2:6.3f}")
