"""
Traffic Demand Prediction  -  Gridlock Hackathon
=================================================

End-to-end regression pipeline that predicts `demand` from spatio-temporal
traffic features.

Pipeline
--------
1. Load train/test.
2. Feature engineering:
      * geohash  -> latitude / longitude (self-contained base-32 decoder)
      * timestamp / day -> calendar & cyclical time features
      * categorical columns -> integer codes (LightGBM-native categoricals)
3. Cross-validated comparison of LightGBM, XGBoost and CatBoost (5-fold R2).
4. Light hyper-parameter tuning of the best performer.
5. Retrain on the full training set, predict test, write submission.csv.
6. Print the final cross-validated R2 score.

Evaluation metric:  score = max(0, 100 * r2_score(actual, predicted))

Run:
    python solution.py
Expects train.csv and test.csv in the working directory (or DATA_DIR).
Outputs submission.csv  (41778 x 2:  Index, demand).
"""

import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------- #
# Configuration
# --------------------------------------------------------------------------- #
DATA_DIR = os.environ.get("DATA_DIR", r"C:\\Users\\ASUS\\OneDrive\\Desktop\\comps\\dataset")
TRAIN_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_PATH = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

TARGET = "demand"
ID_COL = "Index"
N_SPLITS = 5
SEED = 42


# --------------------------------------------------------------------------- #
# Geohash decoding (self-contained, no external dependency)
# --------------------------------------------------------------------------- #
_BASE32 = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE_MAP = {c: i for i, c in enumerate(_BASE32)}


def _decode_geohash(geohash):
    """Decode a geohash string into an approximate (lat, lon) centre point."""
    if not isinstance(geohash, str) or geohash == "":
        return np.nan, np.nan

    lat_lo, lat_hi = -90.0, 90.0
    lon_lo, lon_hi = -180.0, 180.0
    is_even = True  # even bits encode longitude, odd bits latitude

    for ch in geohash.lower():
        cd = _DECODE_MAP.get(ch)
        if cd is None:                      # skip unexpected characters
            continue
        for mask in (16, 8, 4, 2, 1):
            if is_even:
                mid = (lon_lo + lon_hi) / 2.0
                if cd & mask:
                    lon_lo = mid
                else:
                    lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2.0
                if cd & mask:
                    lat_lo = mid
                else:
                    lat_hi = mid
            is_even = not is_even

    return (lat_lo + lat_hi) / 2.0, (lon_lo + lon_hi) / 2.0


def add_geohash_features(df):
    """Add lat / lon columns by decoding the (often repeated) geohash values."""
    if "geohash" not in df.columns:
        return df
    gh = df["geohash"].astype(str)
    # Decode each unique geohash once, then map back -> big speed-up.
    cache = {g: _decode_geohash(g) for g in gh.unique()}
    df["gh_lat"] = gh.map(lambda g: cache[g][0])
    df["gh_lon"] = gh.map(lambda g: cache[g][1])
    return df


# --------------------------------------------------------------------------- #
# Time features
# --------------------------------------------------------------------------- #
def add_time_features(df):
    """Derive calendar and cyclical features from `timestamp` and `day`."""
    if "timestamp" in df.columns:
        ts = df["timestamp"]
        dt = pd.to_datetime(ts, errors="coerce")          # try string/ISO dates

        # Fall back to epoch seconds / ms if the column is purely numeric.
        if dt.isna().all():
            num = pd.to_numeric(ts, errors="coerce")
            if num.notna().any():
                unit = "ms" if num.dropna().median() > 1e11 else "s"
                dt = pd.to_datetime(num, unit=unit, errors="coerce")

        df["ts_hour"] = dt.dt.hour
        df["ts_minute"] = dt.dt.minute
        df["ts_dayofweek"] = dt.dt.dayofweek
        df["ts_day"] = dt.dt.day
        df["ts_month"] = dt.dt.month
        df["ts_is_weekend"] = (dt.dt.dayofweek >= 5).astype("float")

        # Cyclical encodings so the model sees 23:00 and 00:00 as adjacent.
        hod = df["ts_hour"].fillna(0)
        df["hour_sin"] = np.sin(2 * np.pi * hod / 24.0)
        df["hour_cos"] = np.cos(2 * np.pi * hod / 24.0)
        dow = df["ts_dayofweek"].fillna(0)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

    # `day` may be a weekday name, a number, or a date string -> handle all.
    if "day" in df.columns:
        day_num = pd.to_numeric(df["day"], errors="coerce")
        if day_num.notna().any():
            df["day_num"] = day_num
        else:
            day_dt = pd.to_datetime(df["day"], errors="coerce")
            if day_dt.notna().any():
                df["day_num"] = day_dt.dt.dayofweek
            else:
                # weekday names -> ordinal
                names = {n: i for i, n in enumerate(
                    ["monday", "tuesday", "wednesday", "thursday",
                     "friday", "saturday", "sunday"])}
                df["day_num"] = (df["day"].astype(str).str.lower()
                                 .str[:9].map(lambda s: names.get(s, np.nan)))
    return df


# --------------------------------------------------------------------------- #
# Categorical encoding
# --------------------------------------------------------------------------- #
def encode_categoricals(train, test, cat_cols):
    """Integer-encode categoricals consistently across train + test."""
    for col in cat_cols:
        if col not in train.columns:
            continue
        combined = pd.concat([train[col], test[col]], axis=0).astype(str)
        codes = combined.astype("category").cat.codes
        train[col] = codes.iloc[:len(train)].values
        test[col] = codes.iloc[len(train):].values
    return train, test


def build_features(train_raw, test_raw):
    """Run the full feature-engineering pipeline on both splits."""
    train, test = train_raw.copy(), test_raw.copy()

    for df in (train, test):
        add_geohash_features(df)
        add_time_features(df)

    # Columns that are inherently categorical in this dataset.
    candidate_cats = ["geohash", "RoadType", "LargeVehicles",
                      "Landmarks", "Weather"]
    cat_cols = [c for c in candidate_cats if c in train.columns]
    train, test = encode_categoricals(train, test, cat_cols)

    # Final feature list: everything except id, target and raw time strings.
    drop_cols = {ID_COL, TARGET, "timestamp", "day"}
    features = [c for c in train.columns if c not in drop_cols]
    # Keep only columns present in test as well.
    features = [c for c in features if c in test.columns]

    # Make sure every feature is numeric for the boosters.
    for col in features:
        train[col] = pd.to_numeric(train[col], errors="coerce")
        test[col] = pd.to_numeric(test[col], errors="coerce")

    return train, test, features, cat_cols


# --------------------------------------------------------------------------- #
# Model factories
# --------------------------------------------------------------------------- #
def make_lgb():
    return lgb.LGBMRegressor(
        n_estimators=2000, learning_rate=0.03, num_leaves=63,
        subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        reg_lambda=1.0, min_child_samples=30,
        random_state=SEED, n_jobs=-1, verbosity=-1,
    )


def make_xgb():
    return xgb.XGBRegressor(
        n_estimators=2000, learning_rate=0.03, max_depth=8,
        subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
        min_child_weight=5, random_state=SEED, n_jobs=-1,
        tree_method="hist", eval_metric="rmse",
    )


def make_cat():
    return CatBoostRegressor(
        iterations=2000, learning_rate=0.03, depth=8,
        l2_leaf_reg=3.0, random_seed=SEED, loss_function="RMSE",
        verbose=0, allow_writing_files=False,
    )


# --------------------------------------------------------------------------- #
# Cross-validation
# --------------------------------------------------------------------------- #
def cross_validate(model_name, X, y, features):
    """5-fold CV; returns (mean R2, out-of-fold predictions)."""
    kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(X))
    scores = []

    for fold, (tr_idx, va_idx) in enumerate(kf.split(X), 1):
        X_tr, X_va = X.iloc[tr_idx], X.iloc[va_idx]
        y_tr, y_va = y.iloc[tr_idx], y.iloc[va_idx]

        if model_name == "lgb":
            model = make_lgb()
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                      callbacks=[lgb.early_stopping(100, verbose=False)])
        elif model_name == "xgb":
            model = make_xgb()
            model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        else:  # catboost
            model = make_cat()
            model.fit(X_tr, y_tr, eval_set=(X_va, y_va),
                      early_stopping_rounds=100, verbose=False)

        pred = model.predict(X_va)
        oof[va_idx] = pred
        fold_r2 = r2_score(y_va, pred)
        scores.append(fold_r2)
        print(f"    [{model_name}] fold {fold}: R2 = {fold_r2:.5f}")

    mean_r2 = float(np.mean(scores))
    print(f"  -> {model_name} mean CV R2 = {mean_r2:.5f}")
    return mean_r2, oof


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main():
    if not (os.path.exists(TRAIN_PATH) and os.path.exists(TEST_PATH)):
        raise FileNotFoundError(
            f"Could not find train.csv / test.csv in '{DATA_DIR}'. "
            "Download the dataset and place the files there "
            "(or set the DATA_DIR environment variable)."
        )

    print("Loading data ...")
    train_raw = pd.read_csv(TRAIN_PATH)
    test_raw = pd.read_csv(TEST_PATH)
    print(f"  train: {train_raw.shape}   test: {test_raw.shape}")

    # ---- Target handling ------------------------------------------------- #
    # demand is non-negative count-like data; log1p stabilises the variance
    # and tends to improve R2 for skewed traffic demand.
    y_raw = train_raw[TARGET].astype(float)
    use_log = (y_raw.min() >= 0) and (y_raw.skew() > 1.0)
    y = np.log1p(y_raw) if use_log else y_raw
    print(f"  target log1p transform: {use_log}")

    # ---- Feature engineering -------------------------------------------- #
    print("Engineering features ...")
    train, test, features, cat_cols = build_features(train_raw, test_raw)
    print(f"  {len(features)} features: {features}")

    X = train[features]
    X_test = test[features]

    # ---- Model comparison ----------------------------------------------- #
    print("\nCross-validating candidate models ...")
    results = {}
    for name in ("lgb", "xgb", "cat"):
        mean_r2, _ = cross_validate(name, X, y, features)
        results[name] = mean_r2

    best_name = max(results, key=results.get)
    print(f"\nBest model: {best_name}  (CV R2 = {results[best_name]:.5f})")

    # ---- Light hyper-parameter tuning of the best model ----------------- #
    print(f"\nTuning {best_name} ...")
    best_cv, best_params = tune_model(best_name, X, y)
    print(f"  tuned CV R2 = {best_cv:.5f}  params = {best_params}")

    # ---- Retrain on full data & predict test ---------------------------- #
    print("\nRetraining best model on full data and predicting test ...")
    final_model = build_tuned_model(best_name, best_params)
    final_model.fit(X, y)
    test_pred = final_model.predict(X_test)
    if use_log:
        test_pred = np.expm1(test_pred)
    test_pred = np.clip(test_pred, 0, None)   # demand cannot be negative

    # ---- Write submission ----------------------------------------------- #
    submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: test_pred})
    submission.to_csv(SUBMISSION_PATH, index=False)
    print(f"\nSaved {SUBMISSION_PATH}  shape={submission.shape}")

    # ---- Final score ---------------------------------------------------- #
    final_cv = max(best_cv, results[best_name])
    print(f"\n{'='*48}")
    print(f"FINAL CV R2 score: {final_cv:.5f}")
    print(f"Competition score : {max(0, 100 * final_cv):.4f}")
    print(f"{'='*48}")


# --------------------------------------------------------------------------- #
# Hyper-parameter tuning helpers
# --------------------------------------------------------------------------- #
def build_tuned_model(name, params):
    if name == "lgb":
        base = make_lgb().get_params(); base.update(params)
        return lgb.LGBMRegressor(**base)
    if name == "xgb":
        base = make_xgb().get_params(); base.update(params)
        return xgb.XGBRegressor(**base)
    base = make_cat().get_params(); base.update(params)
    return CatBoostRegressor(**base)


def _quick_cv_score(model, X, y):
    """3-fold quick R2 used during tuning (no early stopping for simplicity)."""
    kf = KFold(n_splits=3, shuffle=True, random_state=SEED)
    scores = []
    for tr, va in kf.split(X):
        m = model
        m.fit(X.iloc[tr], y.iloc[tr])
        scores.append(r2_score(y.iloc[va], m.predict(X.iloc[va])))
    return float(np.mean(scores))


def tune_model(name, X, y):
    """Small grid search over the most impactful parameters of the best model."""
    if name == "lgb":
        grid = [
            {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 800},
            {"num_leaves": 63, "learning_rate": 0.03, "n_estimators": 1200},
            {"num_leaves": 127, "learning_rate": 0.02, "n_estimators": 1500},
        ]
    elif name == "xgb":
        grid = [
            {"max_depth": 6, "learning_rate": 0.05, "n_estimators": 800},
            {"max_depth": 8, "learning_rate": 0.03, "n_estimators": 1200},
            {"max_depth": 10, "learning_rate": 0.02, "n_estimators": 1500},
        ]
    else:
        grid = [
            {"depth": 6, "learning_rate": 0.05, "iterations": 800},
            {"depth": 8, "learning_rate": 0.03, "iterations": 1200},
            {"depth": 10, "learning_rate": 0.02, "iterations": 1500},
        ]

    best_score, best_params = -np.inf, grid[0]
    for params in grid:
        model = build_tuned_model(name, params)
        score = _quick_cv_score(model, X, y)
        print(f"    {params} -> R2 = {score:.5f}")
        if score > best_score:
            best_score, best_params = score, params
    return best_score, best_params


if __name__ == "__main__":
    main()
