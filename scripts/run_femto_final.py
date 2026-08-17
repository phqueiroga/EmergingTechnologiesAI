#!/usr/bin/env python3
"""Final FEMTO/PRONOSTIA Green AI comparison.

Target: normalized RUL (fraction of life remaining), within-condition splits
(train on one bearing of a condition, test on its sibling), consistent with
FEMTO/PRONOSTIA literature practice and the diagnostics already run.

Metrics per (strategy, model, condition):
  - MAE / R2 on rul_frac
  - critical recall: of test points truly in the critical zone (frac <= CRITICAL_FRAC),
    fraction correctly flagged
  - lead_time_min: real minutes-to-failure remaining at the first correct alarm
    (chronological), i.e. how much warning the model gives in practice
  - train/infer time, energy (kWh), CO2e (g), model size (KB)
"""
import pathlib
import pickle
import time

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
CRITICAL_FRAC = 0.20  # last 20% of life = "critical zone"
OUT_DIR = pathlib.Path("/home/user/EmergingTechnologiesAI/taba_femto_outputs")
DATA_DIR = pathlib.Path("/workspace/wkzs111/phm-ieee-2012-data-challenge-dataset/Learning_set")

features = pd.read_csv(OUT_DIR / "features_reference.csv")
feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]
features["rul_frac"] = features["rul_min"] / features.groupby("bearing")["rul_min"].transform("max")

CONDITIONS = {
    1: ("Bearing1_1", "Bearing1_2"),
    2: ("Bearing2_1", "Bearing2_2"),
    3: ("Bearing3_1", "Bearing3_2"),
}

FS_NATIVE = 25600
FREQ_BANDS_HZ = [(0, 1000), (1000, 5000), (5000, FS_NATIVE // 2)]
FEATURE_NAMES = [
    "rms", "std", "p2p", "crest", "skew", "kurtosis", "energy",
    "spec_centroid", "spec_entropy", "band_low", "band_mid", "band_high",
]

from scipy import stats
from scipy.fft import rfft, rfftfreq


def read_measurement(path):
    df = pd.read_csv(path, header=None)
    h = df.iloc[:, -2].to_numpy(dtype=float)
    v = df.iloc[:, -1].to_numpy(dtype=float)
    return h, v


def band_energy_ratios(signal, fs, bands):
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1.0 / fs)
    power = spectrum ** 2
    total = power.sum()
    if total <= 0:
        return [0.0 for _ in bands]
    return [float(power[(freqs >= lo) & (freqs < hi)].sum() / total) for lo, hi in bands]


def extract_channel_features(signal, fs):
    signal = signal.astype(float)
    rms = float(np.sqrt(np.mean(signal ** 2)))
    std = float(np.std(signal))
    p2p = float(np.ptp(signal))
    crest = float(np.max(np.abs(signal)) / rms) if rms > 0 else 0.0
    skew = float(stats.skew(signal))
    kurt = float(stats.kurtosis(signal))
    energy = float(np.sum(signal ** 2))
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(len(signal), d=1.0 / fs)
    power = spectrum ** 2
    total_power = power.sum()
    if total_power > 0:
        centroid = float(np.sum(freqs * power) / total_power)
        p_norm = power / total_power
        p_norm = p_norm[p_norm > 0]
        spec_entropy = float(-np.sum(p_norm * np.log2(p_norm)) / np.log2(len(p_norm)))
    else:
        centroid, spec_entropy = 0.0, 0.0
    return [rms, std, p2p, crest, skew, kurt, energy, centroid, spec_entropy] + band_energy_ratios(signal, fs, FREQ_BANDS_HZ)


def rebuild_half_rate(bearing_name, obs_indices):
    bpaths = sorted((DATA_DIR / bearing_name).glob("acc_*.csv"))
    rows = []
    for i in obs_indices:
        h, v = read_measurement(bpaths[i])
        h_ds, v_ds = h[::2], v[::2]
        fs_ds = FS_NATIVE // 2
        h_feats = extract_channel_features(h_ds, fs_ds)
        v_feats = extract_channel_features(v_ds, fs_ds)
        row = {f"h_{n}": val for n, val in zip(FEATURE_NAMES, h_feats)}
        row.update({f"v_{n}": val for n, val in zip(FEATURE_NAMES, v_feats)})
        rows.append(row)
    return pd.DataFrame(rows)[feature_columns]


def top_k_features(X_train, y_train, columns, k=10):
    rf = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE, n_jobs=-1)
    rf.fit(X_train, y_train)
    order = np.argsort(rf.feature_importances_)[::-1]
    return [columns[i] for i in order[:k]]


def get_model(name):
    if name == "Ridge":
        return Ridge(alpha=1.0, random_state=RANDOM_STATE)
    if name == "HistGradientBoosting":
        return HistGradientBoostingRegressor(random_state=RANDOM_STATE)
    if name == "RandomForest":
        return RandomForestRegressor(n_estimators=300, random_state=RANDOM_STATE, n_jobs=-1)
    if name == "MLP":
        return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000, early_stopping=True, random_state=RANDOM_STATE)
    raise ValueError(name)


MODEL_NAMES = ["Ridge", "HistGradientBoosting", "RandomForest", "MLP"]
ESTIMATED_WATTS = 15.0
GRID_CARBON_INTENSITY_G_PER_KWH = 429.0


def measure(fn):
    t0 = time.time()
    result = fn()
    elapsed = time.time() - t0
    energy_kwh = (ESTIMATED_WATTS * elapsed / 3600) / 1000
    co2_g = energy_kwh * GRID_CARBON_INTENSITY_G_PER_KWH
    return result, elapsed, energy_kwh, co2_g


def lead_time_minutes(test_df_sorted, y_pred_frac):
    """Real minutes remaining at the first correctly-flagged critical alarm."""
    truly_critical = test_df_sorted["rul_frac"].to_numpy() <= CRITICAL_FRAC
    if not truly_critical.any():
        return np.nan
    pred_critical = y_pred_frac <= CRITICAL_FRAC
    correct_alarm = truly_critical & pred_critical
    if not correct_alarm.any():
        return 0.0  # never caught it in time
    first_idx = np.argmax(correct_alarm)
    return float(test_df_sorted["rul_min"].to_numpy()[first_idx])


def build_strategy_data(strategy, train_df, test_df, train_bearing, test_bearing):
    if strategy == "reference":
        cols = feature_columns
        return train_df[cols].to_numpy(), test_df[cols].to_numpy(), cols, 1
    if strategy == "half-rate":
        X_train = rebuild_half_rate(train_bearing, train_df["obs_index"]).to_numpy()
        X_test = rebuild_half_rate(test_bearing, test_df["obs_index"]).to_numpy()
        return X_train, X_test, feature_columns, 2
    if strategy == "top-10-features":
        scaler_tmp = StandardScaler().fit(train_df[feature_columns])
        top_cols = top_k_features(scaler_tmp.transform(train_df[feature_columns]),
                                   train_df["rul_frac"].to_numpy(), feature_columns, k=10)
        return train_df[top_cols].to_numpy(), test_df[top_cols].to_numpy(), top_cols, 1
    raise ValueError(strategy)


STRATEGIES = ["reference", "half-rate", "top-10-features"]
results = []

for cond, (train_b, test_b) in CONDITIONS.items():
    train_df = features[features["bearing"] == train_b].sort_values("obs_index").reset_index(drop=True)
    test_df = features[features["bearing"] == test_b].sort_values("obs_index").reset_index(drop=True)

    for strategy in STRATEGIES:
        t0 = time.time()
        X_train_raw, X_test_raw, cols_used, downsample = build_strategy_data(strategy, train_df, test_df, train_b, test_b)
        build_t = time.time() - t0

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        y_train = train_df["rul_frac"].to_numpy()
        y_test = test_df["rul_frac"].to_numpy()

        for model_name in MODEL_NAMES:
            model = get_model(model_name)
            _, train_time, train_energy_kwh, train_co2_g = measure(lambda: model.fit(X_train, y_train))
            y_pred, infer_time, infer_energy_kwh, infer_co2_g = measure(lambda: model.predict(X_test))

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)

            truly_critical = y_test <= CRITICAL_FRAC
            if truly_critical.sum() > 0:
                pred_flagged = y_pred <= CRITICAL_FRAC
                critical_recall = float((pred_flagged & truly_critical).sum() / truly_critical.sum())
            else:
                critical_recall = float("nan")

            lt = lead_time_minutes(test_df, y_pred)
            model_size_kb = len(pickle.dumps(model)) / 1024

            results.append({
                "condition": cond, "train_bearing": train_b, "test_bearing": test_b,
                "strategy": strategy, "model": model_name, "n_features": len(cols_used),
                "downsample_factor": downsample, "mae_frac": mae, "rmse_frac": rmse, "r2": r2,
                "critical_recall": critical_recall, "lead_time_min": lt,
                "data_build_s": build_t, "train_time_s": train_time, "infer_time_s": infer_time,
                "train_energy_kwh": train_energy_kwh, "co2_g": train_co2_g + infer_co2_g,
                "model_size_kb": model_size_kb,
            })
            print(f"cond{cond} {strategy:16s} {model_name:22s} MAE={mae:6.3f} R2={r2:6.3f} "
                  f"recall={critical_recall:.2f} lead_time={lt if lt==lt else float('nan'):7.1f}min "
                  f"energy={train_energy_kwh:.2e}kWh")

results_df = pd.DataFrame(results)
results_df.to_csv(OUT_DIR / "final_results_normalized_rul.csv", index=False)

print("\n=== Aggregated across conditions (mean per strategy x model) ===")
agg = results_df.groupby(["strategy", "model"]).agg(
    mae_frac=("mae_frac", "mean"), r2=("r2", "mean"),
    critical_recall=("critical_recall", "mean"), lead_time_min=("lead_time_min", "mean"),
    train_energy_kwh=("train_energy_kwh", "mean"), co2_g=("co2_g", "mean"),
    model_size_kb=("model_size_kb", "mean"), n_features=("n_features", "first"),
).reset_index()
agg.to_csv(OUT_DIR / "final_results_aggregated.csv", index=False)
print(agg.to_string())
