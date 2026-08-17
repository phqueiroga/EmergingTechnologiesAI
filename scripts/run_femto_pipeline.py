#!/usr/bin/env python3
"""FEMTO/PRONOSTIA RUL Green AI pipeline (headless run for validation)."""
import pathlib
import pickle
import sys
import time

import numpy as np
import pandas as pd
from scipy import stats
from scipy.fft import rfft, rfftfreq

from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import GroupShuffleSplit
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = pathlib.Path(sys.argv[1] if len(sys.argv) > 1 else
                         "/workspace/wkzs111/phm-ieee-2012-data-challenge-dataset/Learning_set")
OUT_DIR = pathlib.Path("/home/user/EmergingTechnologiesAI/taba_femto_outputs")
OUT_DIR.mkdir(exist_ok=True)

FS_NATIVE = 25600
SAMPLE_INTERVAL_S = 10
RUL_CAP_MIN = None
FREQ_BANDS_HZ = [(0, 1000), (1000, 5000), (5000, FS_NATIVE // 2)]
CRITICAL_RUL_MIN = 20
TARGET = "rul_min"

FEATURE_NAMES = [
    "rms", "std", "p2p", "crest", "skew", "kurtosis", "energy",
    "spec_centroid", "spec_entropy", "band_low", "band_mid", "band_high",
]


def read_measurement(path: pathlib.Path):
    df = pd.read_csv(path, header=None)
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all")
    if df.shape[1] < 2:
        raise ValueError(f"Arquivo com menos de duas colunas: {path}")
    h = df.iloc[:, -2].to_numpy(dtype=float)
    v = df.iloc[:, -1].to_numpy(dtype=float)
    return h, v


def band_energy_ratios(signal, fs, bands):
    n = len(signal)
    spectrum = np.abs(rfft(signal))
    freqs = rfftfreq(n, d=1.0 / fs)
    power = spectrum ** 2
    total = power.sum()
    if total <= 0:
        return [0.0 for _ in bands]
    ratios = []
    for lo, hi in bands:
        mask = (freqs >= lo) & (freqs < hi)
        ratios.append(float(power[mask].sum() / total))
    return ratios


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
        centroid = 0.0
        spec_entropy = 0.0

    band_ratios = band_energy_ratios(signal, fs, FREQ_BANDS_HZ)
    return [rms, std, p2p, crest, skew, kurt, energy, centroid, spec_entropy] + band_ratios


def build_feature_table(csv_files, fs):
    by_bearing = {}
    for path in csv_files:
        bearing = path.parent.name
        by_bearing.setdefault(bearing, []).append(path)
    for bearing in by_bearing:
        by_bearing[bearing] = sorted(by_bearing[bearing])

    rows = []
    for bearing, paths in by_bearing.items():
        n_obs = len(paths)
        t0 = time.time()
        for i, path in enumerate(paths):
            h, v = read_measurement(path)
            h_feats = extract_channel_features(h, fs)
            v_feats = extract_channel_features(v, fs)
            rul_min = (n_obs - 1 - i) * SAMPLE_INTERVAL_S / 60
            if RUL_CAP_MIN is not None:
                rul_min = min(rul_min, RUL_CAP_MIN)
            row = {"bearing": bearing, "obs_index": i, "n_obs_bearing": n_obs, "rul_min": rul_min}
            row.update({f"h_{name}": val for name, val in zip(FEATURE_NAMES, h_feats)})
            row.update({f"v_{name}": val for name, val in zip(FEATURE_NAMES, v_feats)})
            rows.append(row)
        print(f"  {bearing}: {n_obs} obs in {time.time() - t0:.1f}s")
    return pd.DataFrame(rows)


def main():
    print("DATA_DIR:", DATA_DIR)
    csv_files = sorted(DATA_DIR.rglob("acc_*.csv"))
    print("Acceleration CSV files found:", len(csv_files))
    if len(csv_files) == 0:
        raise ValueError("Nenhum arquivo acc_*.csv encontrado.")

    sample = pd.read_csv(csv_files[0], header=None)
    print("Sample shape:", sample.shape)
    print(sample.head())

    print("\nExtracting features...")
    t0 = time.time()
    features = build_feature_table(csv_files, FS_NATIVE)
    print(f"Feature extraction took {time.time() - t0:.1f}s total")

    feature_columns = [c for c in features.columns if c.startswith("h_") or c.startswith("v_")]
    print("\nRows:", len(features))
    print("Bearings:", features["bearing"].nunique())
    print("Feature columns:", len(feature_columns))
    print("Total NaNs in features:", features[feature_columns].isna().sum().sum())

    # --- validation gate ---
    errors = []
    if len(feature_columns) != 24:
        errors.append(f"Esperado 24 colunas, encontrado {len(feature_columns)}.")
    if features[feature_columns].isna().sum().sum() > 0:
        errors.append("Existem NaNs na tabela de features.")

    nunique = features[feature_columns].nunique().sort_values()
    n_rows = len(features)
    near_constant = nunique[nunique <= max(1, int(0.01 * n_rows))]
    if len(near_constant) > 0:
        errors.append(f"Features quase constantes: {list(near_constant.index)}")

    zero_fraction = (features[feature_columns] == 0).mean()
    mostly_zero = zero_fraction[zero_fraction > 0.5]
    if len(mostly_zero) > 0:
        errors.append(f"Features majoritariamente zeradas: {list(mostly_zero.index)}")

    h_cols = [c for c in feature_columns if c.startswith("h_")]
    v_cols = [c.replace("h_", "v_", 1) for c in h_cols]
    identical_frac = (features[h_cols].to_numpy() == features[v_cols].to_numpy()).mean()
    if identical_frac > 0.5:
        errors.append(f"Canais h/v idênticos em {identical_frac:.0%} das células.")

    print("\nNunique per feature:")
    print(nunique)

    if errors:
        print("\nVALIDATION GATE FAILED:")
        for e in errors:
            print(" -", e)
        raise SystemExit(1)

    print("\n✅ Validation gate passed.")
    features.to_csv(OUT_DIR / "features_reference.csv", index=False)

    # --- train/test split ---
    splitter = GroupShuffleSplit(n_splits=1, test_size=0.25, random_state=RANDOM_STATE)
    train_idx, test_idx = next(splitter.split(features, groups=features["bearing"]))
    train_df = features.iloc[train_idx].reset_index(drop=True)
    test_df = features.iloc[test_idx].reset_index(drop=True)
    print("\nTrain bearings:", sorted(train_df["bearing"].unique()))
    print("Test bearings:", sorted(test_df["bearing"].unique()))

    def top_k_features_by_importance(X_train, y_train, columns, k=10):
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
            return MLPRegressor(hidden_layer_sizes=(64, 32), max_iter=2000,
                                 early_stopping=True, random_state=RANDOM_STATE)
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

    def rebuild_half_rate(df, all_feature_cols):
        rows = []
        for bearing, group in df.groupby("bearing"):
            bpaths = sorted((DATA_DIR / bearing).glob("acc_*.csv"))
            for i in group["obs_index"]:
                h, v = read_measurement(bpaths[i])
                h_ds, v_ds = h[::2], v[::2]
                fs_ds = FS_NATIVE // 2
                h_feats = extract_channel_features(h_ds, fs_ds)
                v_feats = extract_channel_features(v_ds, fs_ds)
                row = {f"h_{n}": val for n, val in zip(FEATURE_NAMES, h_feats)}
                row.update({f"v_{n}": val for n, val in zip(FEATURE_NAMES, v_feats)})
                rows.append(row)
        return pd.DataFrame(rows)[all_feature_cols]

    def build_strategy_data(strategy, train_df, test_df, all_feature_cols):
        if strategy == "reference":
            cols = all_feature_cols
            return train_df[cols].to_numpy(), test_df[cols].to_numpy(), cols, 1
        if strategy == "half-rate":
            X_train = rebuild_half_rate(train_df, all_feature_cols).to_numpy()
            X_test = rebuild_half_rate(test_df, all_feature_cols).to_numpy()
            return X_train, X_test, all_feature_cols, 2
        if strategy == "top-10-features":
            scaler_tmp = StandardScaler().fit(train_df[all_feature_cols])
            top_cols = top_k_features_by_importance(
                scaler_tmp.transform(train_df[all_feature_cols]),
                train_df[TARGET].to_numpy(), all_feature_cols, k=10)
            return train_df[top_cols].to_numpy(), test_df[top_cols].to_numpy(), top_cols, 1
        raise ValueError(strategy)

    STRATEGIES = ["reference", "half-rate", "top-10-features"]
    results = []

    for strategy in STRATEGIES:
        print(f"\n=== Strategy: {strategy} ===")
        t_strat = time.time()
        X_train_raw, X_test_raw, cols_used, downsample = build_strategy_data(
            strategy, train_df, test_df, feature_columns)
        print(f"  data build took {time.time() - t_strat:.1f}s")

        scaler = StandardScaler().fit(X_train_raw)
        X_train = scaler.transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)
        y_train = train_df[TARGET].to_numpy()
        y_test = test_df[TARGET].to_numpy()

        for model_name in MODEL_NAMES:
            model = get_model(model_name)
            _, train_time, train_energy_kwh, train_co2_g = measure(lambda: model.fit(X_train, y_train))
            y_pred, infer_time, infer_energy_kwh, infer_co2_g = measure(lambda: model.predict(X_test))

            mae = mean_absolute_error(y_test, y_pred)
            rmse = mean_squared_error(y_test, y_pred) ** 0.5
            r2 = r2_score(y_test, y_pred)

            critical_mask = y_test <= CRITICAL_RUL_MIN
            if critical_mask.sum() > 0:
                pred_flagged = y_pred <= CRITICAL_RUL_MIN
                critical_recall = float((pred_flagged & critical_mask).sum() / critical_mask.sum())
            else:
                critical_recall = float("nan")

            model_size_kb = len(pickle.dumps(model)) / 1024

            results.append({
                "strategy": strategy, "model": model_name, "n_features": len(cols_used),
                "downsample_factor": downsample, "mae_min": mae, "rmse_min": rmse, "r2": r2,
                "critical_recall": critical_recall, "train_time_s": train_time,
                "infer_time_s": infer_time, "train_energy_kwh": train_energy_kwh,
                "co2_g": train_co2_g + infer_co2_g, "model_size_kb": model_size_kb,
            })
            print(f"  {model_name:22s} MAE={mae:8.2f} min  R2={r2:6.3f}  "
                  f"recall_crit={critical_recall:.2f}  train_t={train_time:.2f}s")

    results_df = pd.DataFrame(results)

    def is_pareto_efficient(costs):
        is_efficient = np.ones(costs.shape[0], dtype=bool)
        for i, c in enumerate(costs):
            if is_efficient[i]:
                is_efficient[is_efficient] = (np.any(costs[is_efficient] < c, axis=1) |
                                               np.all(costs[is_efficient] == c, axis=1))
                is_efficient[i] = True
        return is_efficient

    pareto_costs = results_df[["mae_min", "train_energy_kwh"]].to_numpy()
    results_df["pareto_optimal"] = is_pareto_efficient(pareto_costs)

    results_df[results_df["strategy"] == "reference"].to_csv(OUT_DIR / "reference_results.csv", index=False)
    results_df[results_df["strategy"] != "reference"].to_csv(OUT_DIR / "green_strategy_results.csv", index=False)
    results_df.to_csv(OUT_DIR / "all_results_pareto.csv", index=False)

    print("\nAll results:")
    print(results_df.to_string())
    print("\nSaved outputs to", OUT_DIR)


if __name__ == "__main__":
    main()
