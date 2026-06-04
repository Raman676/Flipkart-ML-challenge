#!/usr/bin/env python3
"""Generate updated_nb.ipynb with all Phase 1-6 improvements."""

import json


def make_source(code):
    """Convert a multi-line string to notebook source format (list of lines)."""
    if code.startswith('\n'):
        code = code[1:]
    if code.endswith('\n'):
        code = code[:-1]
    lines = code.split('\n')
    result = []
    for i, line in enumerate(lines):
        if i < len(lines) - 1:
            result.append(line + '\n')
        else:
            result.append(line)
    return result


cells = []

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 1: Imports & Configuration
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "967aa51a",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
import os
import warnings

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.metrics import r2_score
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.isotonic import IsotonicRegression

import lightgbm as lgb
import xgboost as xgb
from catboost import CatBoostRegressor
import optuna
from optuna.samplers import CmaEsSampler

warnings.filterwarnings('ignore')

# \u2500\u2500 Configuration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
DATA_DIR        = os.environ.get("DATA_DIR", ".")
TRAIN_PATH      = os.path.join(DATA_DIR, "train.csv")
TEST_PATH       = os.path.join(DATA_DIR, "test.csv")
SUBMISSION_PATH = "submission.csv"

TARGET   = "demand"
ID_COL   = "Index"
N_SPLITS = 10
SEED     = 42
MULTI_SEEDS = [42, 123, 2024]
print("Imports OK")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 2: Geohash utilities (FIXED _GH_NEIGHBOUR + added neighbour function)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "6e34fa68",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Geohash utilities \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
_BASE32    = "0123456789bcdefghjkmnpqrstuvwxyz"
_DECODE_MAP = {c: i for i, c in enumerate(_BASE32)}

# FIXED: Corrected neighbour direction offsets (removed corrupted strings)
_GH_NEIGHBOUR = {
    'n': {'even': 'p0r21436x8zb9dcf5h7kjnmqesgutwvy', 'odd': 'bc01fg45238967deuvhjyznpkmstqrwx'},
    's': {'even': '14365h7k9dcfesgujnmqp0r2twvyx8zb', 'odd': '238967debc01fg45uvhjyznpkmstqrwx'},
    'e': {'even': 'bc01fg45238967deuvhjyznpkmstqrwx', 'odd': 'p0r21436x8zb9dcf5h7kjnmqesgutwvy'},
    'w': {'even': '238967debc01fg45uvhjyznpkmstqrwx', 'odd': '14365h7k9dcfesgujnmqp0r2twvyx8zb'},
}

_GH_BORDER = {
    'n': {'even': 'prxz', 'odd': 'bcfguvyz'},
    's': {'even': '028b', 'odd': '0145hjnp'},
    'e': {'even': 'bcfguvyz', 'odd': 'prxz'},
    'w': {'even': '0145hjnp', 'odd': '028b'},
}


def _decode_geohash(geohash_str):
    \"\"\"Decode a geohash string into (latitude, longitude) centre point.\"\"\"
    if not isinstance(geohash_str, str) or geohash_str == "":
        return np.nan, np.nan
    lat_lo, lat_hi = -90.0,  90.0
    lon_lo, lon_hi = -180.0, 180.0
    is_even = True
    for ch in geohash_str.lower():
        cd = _DECODE_MAP.get(ch)
        if cd is None:
            continue
        for mask in (16, 8, 4, 2, 1):
            if is_even:
                mid = (lon_lo + lon_hi) / 2.0
                if cd & mask: lon_lo = mid
                else:         lon_hi = mid
            else:
                mid = (lat_lo + lat_hi) / 2.0
                if cd & mask: lat_lo = mid
                else:         lat_hi = mid
            is_even = not is_even
    return (lat_lo + lat_hi) / 2.0, (lon_lo + lon_hi) / 2.0


def geohash_neighbour(geohash_str, direction):
    \"\"\"Compute the adjacent geohash in a given direction ('n','s','e','w').\"\"\"
    if not geohash_str or not isinstance(geohash_str, str):
        return None
    gh = geohash_str.lower()
    last_char = gh[-1]
    parent = gh[:-1]
    parity = 'even' if len(gh) % 2 == 1 else 'odd'

    if last_char in _GH_BORDER[direction][parity] and parent:
        parent = geohash_neighbour(parent, direction)
        if parent is None:
            return None

    idx = _GH_NEIGHBOUR[direction][parity].index(last_char)
    return parent + _BASE32[idx]

print("Geohash utils OK")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 3: Day parsing (unchanged)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "1cc88109",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Day parsing \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def parse_day_column(series):
    day_num = pd.to_numeric(series, errors="coerce")
    if day_num.notna().any():
        return (day_num % 7).astype(float)
    day_dt = pd.to_datetime(series, errors="coerce")
    if day_dt.notna().any():
        return day_dt.dt.dayofweek.astype(float)
    name_map = {
        "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
        "friday": 4, "saturday": 5, "sunday": 6,
    }
    return series.astype(str).str.strip().str.lower().map(name_map).astype(float)

print("Day parsing OK")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 4: Build base features (Phase 2 changes)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "1818437e",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Base feature engineering \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
def build_base_features(train_raw, test_raw):
    train = train_raw.copy()
    test  = test_raw.copy()

    for df in (train, test):

        # \u2500\u2500 Spatial: decode geohash \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if "geohash" in df.columns:
            gh = df["geohash"].astype(str)
            cache = {g: _decode_geohash(g) for g in gh.unique()}
            df["gh_lat"]          = gh.map(lambda g: cache[g][0])
            df["gh_lon"]          = gh.map(lambda g: cache[g][1])
            df["geohash_prefix_4"] = gh.str[:4]
            df["geohash_prefix_5"] = gh.str[:5]
            df["geohash_prefix_3"] = gh.str[:3]
            df["geohash_len"]      = gh.str.len().astype(int)

        # \u2500\u2500 Intelligent imputation: RoadType (geohash-mode \u2192 global-mode) \u2500\u2500\u2500
        if "RoadType" in df.columns:
            _rt_global = df["RoadType"].mode()
            _rt_global = _rt_global.iloc[0] if len(_rt_global) > 0 else "Unknown"
            if "geohash" in df.columns:
                _rt_gh_mode = df.groupby("geohash")["RoadType"].transform(
                    lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else np.nan
                )
                df["RoadType"] = df["RoadType"].fillna(_rt_gh_mode)
            df["RoadType"] = df["RoadType"].fillna(_rt_global)

        # \u2500\u2500 Temporal: parse timestamp \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if "timestamp" in df.columns:
            ts = df["timestamp"]
            dt = pd.to_datetime(ts, errors="coerce")
            if dt.isna().all():
                num = pd.to_numeric(ts, errors="coerce")
                if num.notna().any():
                    unit = "ms" if num.dropna().median() > 1e11 else "s"
                    dt   = pd.to_datetime(num, unit=unit, errors="coerce")
            if dt.isna().all():
                parts = ts.astype(str).str.split(":", expand=True)
                if parts.shape[1] >= 2:
                    hours   = pd.to_numeric(parts[0], errors="coerce").fillna(0)
                    minutes = pd.to_numeric(parts[1], errors="coerce").fillna(0)
                    df["ts_hour"]      = hours.astype(int)
                    df["ts_minute"]    = minutes.astype(int)
                    df["ts_dayofweek"] = np.nan
                    df["ts_day"]       = np.nan
                    df["ts_month"]     = np.nan
                else:
                    for c in ["ts_hour","ts_minute","ts_dayofweek","ts_day","ts_month"]:
                        df[c] = np.nan
            else:
                df["ts_hour"]      = dt.dt.hour
                df["ts_minute"]    = dt.dt.minute
                df["ts_dayofweek"] = dt.dt.dayofweek
                df["ts_day"]       = dt.dt.day
                df["ts_month"]     = dt.dt.month

        # \u2500\u2500 Unified 96-slot time index & cyclical encoding \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        hod = df["ts_hour"].fillna(0).astype(float)
        mod = df["ts_minute"].fillna(0).astype(float)
        df["time_slot"] = (hod * 4 + mod // 15).astype(int)
        df["slot_sin"]  = np.sin(2 * np.pi * df["time_slot"] / 96.0)
        df["slot_cos"]  = np.cos(2 * np.pi * df["time_slot"] / 96.0)

        # \u2500\u2500 Intelligent imputation: Temperature (time_slot mean \u2192 global) \u2500\u2500
        if "Temperature" in df.columns:
            df["Temperature"] = pd.to_numeric(df["Temperature"], errors="coerce")
            _slot_temp_mean = df.groupby("time_slot")["Temperature"].transform("mean")
            df["Temperature"] = df["Temperature"].fillna(_slot_temp_mean)
            df["Temperature"] = df["Temperature"].fillna(df["Temperature"].mean())

        if "day" in df.columns:
            df["day_num"] = parse_day_column(df["day"])
            if df["ts_dayofweek"].isna().all():
                df["ts_dayofweek"] = df["day_num"]
        else:
            df["day_num"] = df.get("ts_dayofweek", pd.Series(np.nan, index=df.index))

        dow = df["ts_dayofweek"].fillna(0).astype(float)
        df["dow_sin"] = np.sin(2 * np.pi * dow / 7.0)
        df["dow_cos"] = np.cos(2 * np.pi * dow / 7.0)

        # \u2500\u2500 Finer time-of-day bins \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        hour_vals = df["ts_hour"].fillna(-1).astype(int)
        bins  = [-1,5,8,11,14,17,20,23]
        df["time_segment"] = pd.cut(hour_vals, bins=bins, labels=False).fillna(0).astype(int)

        # \u2500\u2500 Binary flags \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        df["is_weekend"]  = (df["ts_dayofweek"].fillna(0) >= 5).astype(int)
        df["rush_hour"]   = hour_vals.isin({7, 8, 9, 17, 18, 19}).astype(int)
        df["night_flag"]  = ((hour_vals >= 22) | (hour_vals <= 5)).astype(int)
        df["is_monday"]   = (df["ts_dayofweek"].fillna(-1) == 0).astype(int)
        df["is_friday"]   = (df["ts_dayofweek"].fillna(-1) == 4).astype(int)
        df["is_workday_rush"] = (
            (df["is_weekend"] == 0) & (df["rush_hour"] == 1)
        ).astype(int)

        # \u2500\u2500 Binary / numeric cleanup \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if "LargeVehicles" in df.columns:
            lv = df["LargeVehicles"].astype(str).str.strip().str.lower()
            df["LargeVehicles"] = lv.map(
                lambda x: 1 if x in ("1", "yes", "true", "allowed") else 0
            ).astype(int)
        if "Landmarks" in df.columns:
            lm = df["Landmarks"].astype(str).str.strip().str.lower()
            df["Landmarks"] = lm.map(
                lambda x: 1 if x in ("1", "yes", "true") else 0
            ).astype(int)

        # \u2500\u2500 Interaction features \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
        if "NumberofLanes" in df.columns and "Temperature" in df.columns:
            lanes = pd.to_numeric(df["NumberofLanes"], errors="coerce").fillna(0)
            temp  = pd.to_numeric(df["Temperature"],   errors="coerce").fillna(0)
            df["lanes_x_temp"]  = lanes * temp
            df["lanes_squared"] = lanes ** 2
            df["lanes_x_rush"]  = lanes * df["rush_hour"]
            df["lanes_x_wknd"]  = lanes * df["is_weekend"]
            df["temp_squared"]  = temp ** 2

        if "gh_lat" in df.columns:
            df["lat_x_slotsin"] = df["gh_lat"] * df["slot_sin"]
            df["lon_x_slotcos"] = df["gh_lon"] * df["slot_cos"]
            df["lat_x_dow"]     = df["gh_lat"] * dow
            df["lon_x_dow"]     = df["gh_lon"] * dow

        df["hour_x_dow"] = hod * dow

    # \u2500\u2500 Spatial Clustering (K-Means) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if "gh_lat" in train.columns and "gh_lon" in train.columns:
        coords_train = train[["gh_lat", "gh_lon"]].dropna()
        if not coords_train.empty:
            kmeans = KMeans(n_clusters=50, random_state=SEED, n_init=10)
            train.loc[coords_train.index, "spatial_cluster"] = kmeans.fit_predict(coords_train)

            coords_test = test[["gh_lat", "gh_lon"]].dropna()
            if not coords_test.empty:
                test.loc[coords_test.index, "spatial_cluster"] = kmeans.predict(coords_test)

            train["cluster_x_hour"] = train["spatial_cluster"].astype(str) + "_" + train["ts_hour"].astype(str)
            test["cluster_x_hour"] = test["spatial_cluster"].astype(str) + "_" + test["ts_hour"].astype(str)

    # \u2500\u2500 Geohash frequency encoding \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if "geohash" in train.columns:
        gh_counts  = train["geohash"].value_counts().to_dict()
        p3_counts  = train["geohash_prefix_3"].value_counts().to_dict()
        train["gh_freq"]   = train["geohash"].map(gh_counts).fillna(0).astype(int)
        test["gh_freq"]    = test["geohash"].map(gh_counts).fillna(0).astype(int)
        train["gh3_freq"]  = train["geohash_prefix_3"].map(p3_counts).fillna(0).astype(int)
        test["gh3_freq"]   = test["geohash_prefix_3"].map(p3_counts).fillna(0).astype(int)
        rank_map = {k: r for r, k in enumerate(
            sorted(gh_counts, key=gh_counts.get, reverse=True), 1
        )}
        train["gh_rank"] = train["geohash"].map(rank_map).fillna(len(rank_map)+1).astype(int)
        test["gh_rank"]  = test["geohash"].map(rank_map).fillna(len(rank_map)+1).astype(int)

    # \u2500\u2500 Temperature binning \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
    if "Temperature" in train.columns:
        train_temp = pd.to_numeric(train["Temperature"], errors="coerce")
        test_temp  = pd.to_numeric(test["Temperature"],  errors="coerce")
        train["Temperature"] = train_temp
        test["Temperature"]  = test_temp
        try:
            _, bin_edges = pd.qcut(train_temp.dropna(), q=10,
                                   retbins=True, duplicates="drop")
            train["temp_bin"] = pd.cut(train_temp, bins=bin_edges,
                                       labels=False, include_lowest=True)
            test["temp_bin"]  = pd.cut(test_temp,  bins=bin_edges,
                                       labels=False, include_lowest=True)
        except ValueError:
            train["temp_bin"] = 0
            test["temp_bin"]  = 0

    if "NumberofLanes" in train.columns:
        train["NumberofLanes"] = pd.to_numeric(train["NumberofLanes"], errors="coerce")
        test["NumberofLanes"]  = pd.to_numeric(test["NumberofLanes"],  errors="coerce")

    return train, test

print("Feature engineering function defined")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 5: Target encoding (Phase 4: mean, std, median + new keys)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "e9ddc3f9",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Target encoding config \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
TARGET_ENCODE_KEYS = [
    (["geohash"],                          "te_geohash"),
    (["RoadType"],                         "te_RoadType"),
    (["Weather"],                          "te_Weather"),
    (["geohash_prefix_3"],                 "te_geohash_prefix_3"),
    (["geohash_prefix_4"],                 "te_geohash_prefix_4"),
    (["geohash_prefix_5"],                 "te_geohash_prefix_5"),
    (["spatial_cluster"],                  "te_spatial_cluster"),
    (["spatial_cluster", "ts_hour"],       "te_cluster_x_hour"),
    (["geohash", "ts_hour"],               "agg_gh_hour"),
    (["geohash", "day_num"],               "agg_gh_day"),
    (["geohash", "time_segment"],          "agg_gh_segment"),
    (["RoadType", "ts_hour"],              "agg_rt_hour"),
    (["RoadType", "is_weekend"],           "agg_rt_wknd"),
    (["ts_hour",  "is_weekend"],           "agg_hour_wknd"),
    (["Weather",  "RoadType"],             "agg_weather_rt"),
    (["Weather",  "ts_hour"],              "agg_weather_hour"),
    (["geohash", "is_weekend"],            "agg_gh_wknd"),
    (["geohash_prefix_4", "ts_hour"],      "agg_gh4_hour"),
    # NEW: 3-way interaction and time_slot grouping
    (["geohash", "ts_hour", "day_num"],    "agg_gh_hour_day"),
    (["geohash", "time_slot"],             "agg_gh_slot"),
]

SMOOTHING = 10


def _make_group_key(df, cols):
    if len(cols) == 1:
        return df[cols[0]].astype(str)
    return df[cols].astype(str).apply("_".join, axis=1)


def target_encode_oof(train, test, y, all_folds):
    \"\"\"
    OOF smoothed target encoding: computes mean, std, median per group.
    \"\"\"
    global_mean_full   = y.mean()
    global_std_full    = y.std()
    global_median_full = y.median()

    for group_cols, feat_name in TARGET_ENCODE_KEYS:
        missing = [c for c in group_cols if c not in train.columns]
        if missing:
            print(f"  [TE] Skipping {feat_name}: missing columns {missing}")
            train[feat_name]            = global_mean_full
            test[feat_name]             = global_mean_full
            train[feat_name + "_std"]   = global_std_full
            test[feat_name + "_std"]    = global_std_full
            train[feat_name + "_med"]   = global_median_full
            test[feat_name + "_med"]    = global_median_full
            continue

        train_key = _make_group_key(train, group_cols)
        test_key  = _make_group_key(test,  group_cols)

        enc_mean   = np.full(len(train), np.nan, dtype=np.float64)
        enc_std    = np.full(len(train), np.nan, dtype=np.float64)
        enc_median = np.full(len(train), np.nan, dtype=np.float64)

        # ---- OOF encoding (vectorized per fold) ---
        for tr_idx, va_idx in all_folds:
            fold_y = y.iloc[tr_idx]
            fold_global_mean   = fold_y.mean()
            fold_global_std    = fold_y.std()
            fold_global_median = fold_y.median()

            tmp = pd.DataFrame({"key": train_key.iloc[tr_idx], "y": fold_y})
            stats = tmp.groupby("key")["y"].agg(["sum", "count", "std", "median"])
            stats["std"] = stats["std"].fillna(0)

            va_keys = train_key.iloc[va_idx]
            va_cnt  = va_keys.map(stats["count"]).fillna(0)

            # Mean
            va_sum  = va_keys.map(stats["sum"]).fillna(0)
            fold_mean = (va_sum / va_cnt.replace(0, np.nan)).fillna(fold_global_mean)
            enc_mean[va_idx] = (
                (va_cnt * fold_mean + fold_global_mean * SMOOTHING) / (va_cnt + SMOOTHING)
            ).values

            # Std
            va_std = va_keys.map(stats["std"]).fillna(0)
            enc_std[va_idx] = (
                (va_cnt * va_std + fold_global_std * SMOOTHING) / (va_cnt + SMOOTHING)
            ).values

            # Median
            va_med = va_keys.map(stats["median"]).fillna(fold_global_median)
            enc_median[va_idx] = (
                (va_cnt * va_med + fold_global_median * SMOOTHING) / (va_cnt + SMOOTHING)
            ).values

        train[feat_name]          = enc_mean
        train[feat_name + "_std"] = enc_std
        train[feat_name + "_med"] = enc_median

        # ---- Full-data encoding for test ---
        tmp_full = pd.DataFrame({"key": train_key, "y": y})
        full_stats = tmp_full.groupby("key")["y"].agg(["sum", "count", "std", "median"])
        full_stats["std"] = full_stats["std"].fillna(0)

        te_cnt = test_key.map(full_stats["count"]).fillna(0)

        # Mean
        te_sum  = test_key.map(full_stats["sum"]).fillna(0)
        te_mean = (te_sum / te_cnt.replace(0, np.nan)).fillna(global_mean_full)
        test[feat_name] = (
            (te_cnt * te_mean + global_mean_full * SMOOTHING) / (te_cnt + SMOOTHING)
        ).values

        # Std
        te_std = test_key.map(full_stats["std"]).fillna(0)
        test[feat_name + "_std"] = (
            (te_cnt * te_std + global_std_full * SMOOTHING) / (te_cnt + SMOOTHING)
        ).values

        # Median
        te_med = test_key.map(full_stats["median"]).fillna(global_median_full)
        test[feat_name + "_med"] = (
            (te_cnt * te_med + global_median_full * SMOOTHING) / (te_cnt + SMOOTHING)
        ).values

        # NaN fill
        train[feat_name]          = train[feat_name].fillna(global_mean_full)
        test[feat_name]           = test[feat_name].fillna(global_mean_full)
        train[feat_name + "_std"] = train[feat_name + "_std"].fillna(global_std_full)
        test[feat_name + "_std"]  = test[feat_name + "_std"].fillna(global_std_full)
        train[feat_name + "_med"] = train[feat_name + "_med"].fillna(global_median_full)
        test[feat_name + "_med"]  = test[feat_name + "_med"].fillna(global_median_full)

    return train, test

print("Target encoding function defined")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 6: Optuna main tuning (Phase 6: CmaEsSampler, n_trials=50, expanded)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "55b3cd89",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 OPTUNA HYPERPARAMETER TUNING (Phase 6 Upgrades) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Starting Optuna Hyperparameter Tuning (CMA-ES, 50 trials)...")


def objective_lgb(trial):
    params = {
        'objective': 'poisson',
        'device': 'cpu',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'num_leaves': trial.suggest_int('num_leaves', 31, 255),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'min_child_samples': trial.suggest_int('min_child_samples', 10, 50),
        'min_split_gain': trial.suggest_float('min_split_gain', 0.0, 1.0),
        'max_bin': trial.suggest_int('max_bin', 63, 255),
        'bagging_freq': trial.suggest_int('bagging_freq', 1, 10),
        'random_state': SEED,
        'n_jobs': -1,
        'verbosity': -1,
    }
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in all_folds:
        X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
        model = lgb.LGBMRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)],
                  callbacks=[lgb.early_stopping(50, verbose=False)])
        oof_preds[va_idx] = model.predict(X_va)
    return r2_score(y, oof_preds)


def objective_xgb(trial):
    params = {
        'objective': 'count:poisson',
        'eval_metric': 'rmse',
        'device': 'cuda',
        'tree_method': 'hist',
        'n_estimators': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_depth': trial.suggest_int('max_depth', 5, 12),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
        'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
        'min_child_weight': trial.suggest_int('min_child_weight', 1, 7),
        'random_state': SEED,
        'n_jobs': -1,
        'early_stopping_rounds': 50,
    }
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in all_folds:
        X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
        model = xgb.XGBRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
        oof_preds[va_idx] = model.predict(X_va)
    return r2_score(y, oof_preds)


def objective_cat(trial):
    params = {
        'loss_function': 'Poisson',
        'task_type': 'GPU',
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False,
    }
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in all_folds:
        X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        oof_preds[va_idx] = model.predict(X_va)
    return r2_score(y, oof_preds)


def objective_hist(trial):
    params = {
        'loss': 'poisson',
        'max_iter': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 31, 255),
        'min_samples_leaf': trial.suggest_int('min_samples_leaf', 10, 50),
        'l2_regularization': trial.suggest_float('l2_regularization', 1e-3, 10.0, log=True),
        'max_bins': trial.suggest_int('max_bins', 63, 255),
        'random_state': SEED,
        'early_stopping': True,
        'validation_fraction': 0.1,
        'n_iter_no_change': 50,
    }
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in all_folds:
        X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
        model = HistGradientBoostingRegressor(**params)
        model.fit(X_tr, y_tr)
        oof_preds[va_idx] = model.predict(X_va)
    return r2_score(y, oof_preds)


# \u2500\u2500 RUN STUDIES with CMA-ES Sampler \u2500\u2500

print("\\n[1/4] Running XGBoost Tuning (GPU)...")
study_xgb = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=SEED))
study_xgb.optimize(objective_xgb, n_trials=50)
print(f"  [XGBoost]  Best R\u00b2: {study_xgb.best_value:.5f}")

print("\\n[2/4] Running CatBoost Tuning (GPU)...")
study_cat = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=SEED))
study_cat.optimize(objective_cat, n_trials=50)
print(f"  [CatBoost] Best R\u00b2: {study_cat.best_value:.5f}")

print("\\n[3/4] Running LightGBM Tuning (CPU)...")
study_lgb = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=SEED))
study_lgb.optimize(objective_lgb, n_trials=50)
print(f"  [LightGBM] Best R\u00b2: {study_lgb.best_value:.5f}")

print("\\n[4/4] Running HistGBR Tuning (CPU)...")
study_hist = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=SEED))
study_hist.optimize(objective_hist, n_trials=50)
print(f"  [HistGBR]  Best R\u00b2: {study_hist.best_value:.5f}\\n")

best_lgb_params  = study_lgb.best_params
best_xgb_params  = study_xgb.best_params
best_cat_params  = study_cat.best_params
best_hist_params = study_hist.best_params
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 7: Extract and print best parameters
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "c97b243a",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 EXTRACT AND PRINT BEST PARAMETERS \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Copy and paste these dictionaries to skip tuning next time:\\n")
print(f"BEST_XGB_PARAMS = {best_xgb_params}\\n")
print(f"BEST_LGB_PARAMS = {best_lgb_params}\\n")
print(f"BEST_CAT_PARAMS = {best_cat_params}\\n")
print(f"BEST_HIST_PARAMS = {best_hist_params}")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 8: Isolated CatBoost tuning (expanded search space)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "08500e72",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 OPTUNA: CATBOOST ISOLATED TUNING (expanded) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
optuna.logging.set_verbosity(optuna.logging.WARNING)

print("Running Isolated CatBoost Tuning (GPU, CMA-ES, 50 trials)...")

def objective_cat_iso(trial):
    params = {
        'loss_function': 'Poisson',
        'task_type': 'GPU',
        'iterations': 2000,
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.05, log=True),
        'depth': trial.suggest_int('depth', 5, 10),
        'l2_leaf_reg': trial.suggest_float('l2_leaf_reg', 1.0, 10.0, log=True),
        'bagging_temperature': trial.suggest_float('bagging_temperature', 0.0, 10.0),
        'random_strength': trial.suggest_float('random_strength', 0.0, 10.0),
        'random_seed': SEED,
        'verbose': 0,
        'allow_writing_files': False,
    }
    oof_preds = np.zeros(len(X_train))
    for tr_idx, va_idx in all_folds:
        X_tr, y_tr = X_train.iloc[tr_idx], y.iloc[tr_idx]
        X_va, y_va = X_train.iloc[va_idx], y.iloc[va_idx]
        model = CatBoostRegressor(**params)
        model.fit(X_tr, y_tr, eval_set=(X_va, y_va), early_stopping_rounds=50, verbose=0)
        oof_preds[va_idx] = model.predict(X_va)
    return r2_score(y, oof_preds)

study_cat = optuna.create_study(direction="maximize", sampler=CmaEsSampler(seed=SEED))
study_cat.optimize(objective_cat_iso, n_trials=50)
best_cat_params = study_cat.best_params
print(f"  [CatBoost] Best R\u00b2: {study_cat.best_value:.5f}")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 9: Print CatBoost params
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "336e2e58",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
print("\\nBEST_CAT_PARAMS = {")
for k, v in best_cat_params.items():
    print(f"    '{k}': {v},")
print("}")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 10: Locked hyperparameters (+ HistGBR defaults)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "ba4982d9",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 LOCKED HYPERPARAMETERS (OPTUNA OUTPUTS) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Loading hardcoded Optuna parameters...")

BEST_XGB_PARAMS = {
    'learning_rate': 0.0391036011607272,
    'max_depth': 8,
    'subsample': 0.6986387161134314,
    'colsample_bytree': 0.998606250451527,
    'reg_lambda': 0.0131724288267514,
    'reg_alpha': 0.3026346163552876,
    'min_child_weight': 3,
}

BEST_LGB_PARAMS = {
    'learning_rate': 0.01017571722637329,
    'num_leaves': 252,
    'subsample': 0.7496485199184416,
    'colsample_bytree': 0.6239084129638355,
    'reg_lambda': 0.04372689467034487,
    'reg_alpha': 0.028031345287446902,
    'min_child_samples': 22,
}

BEST_CAT_PARAMS = {
    'learning_rate': 0.02103467497167649,
    'depth': 9,
    'l2_leaf_reg': 9.833850185625893,
}

# Reasonable defaults for HistGBR \u2013 re-tune with Optuna for best results
BEST_HIST_PARAMS = {
    'learning_rate': 0.02,
    'max_leaf_nodes': 255,
    'min_samples_leaf': 20,
    'l2_regularization': 1.0,
    'max_bins': 255,
}
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 11: Model definitions (Phase 5: +Hist, 3-seed, seed param)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "4d8e29fa",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Model definitions (Optuna Best Params + GPU + 3-Seed Averaging) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

lgb_params = {
    'objective': 'poisson',
    'n_estimators': 4000,
    'random_state': SEED,
    'n_jobs': -1,
    'verbosity': -1,
    **BEST_LGB_PARAMS,
}

xgb_params = {
    'objective': 'count:poisson',
    'eval_metric': 'rmse',
    'device': 'cuda',
    'tree_method': 'hist',
    'n_estimators': 4000,
    'random_state': SEED,
    'n_jobs': -1,
    'early_stopping_rounds': 150,
    **BEST_XGB_PARAMS,
}

cat_params = {
    'loss_function': 'Poisson',
    'task_type': 'GPU',
    'iterations': 4000,
    'random_seed': SEED,
    'verbose': 0,
    'allow_writing_files': False,
    **BEST_CAT_PARAMS,
}

hist_params = {
    'loss': 'poisson',
    'max_iter': 4000,
    'random_state': SEED,
    'early_stopping': True,
    'validation_fraction': 0.1,
    'n_iter_no_change': 50,
    **BEST_HIST_PARAMS,
}


def train_lgb_fold(X_tr, y_tr, X_va, y_va, seed=SEED):
    params = {**lgb_params, 'random_state': seed}
    model = lgb.LGBMRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        callbacks=[lgb.early_stopping(150, verbose=False),
                   lgb.log_evaluation(-1)],
    )
    return model


def train_xgb_fold(X_tr, y_tr, X_va, y_va, seed=SEED):
    params = {**xgb_params, 'random_state': seed}
    model = xgb.XGBRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_va, y_va)],
        verbose=False,
    )
    return model


def train_cat_fold(X_tr, y_tr, X_va, y_va, seed=SEED):
    params = {**cat_params, 'random_seed': seed}
    model = CatBoostRegressor(**params)
    model.fit(
        X_tr, y_tr,
        eval_set=(X_va, y_va),
        early_stopping_rounds=150,
        verbose=0,
    )
    return model


def train_hist_fold(X_tr, y_tr, X_va, y_va, seed=SEED):
    params = {**hist_params, 'random_state': seed}
    model = HistGradientBoostingRegressor(**params)
    model.fit(X_tr, y_tr)
    return model


def run_model_cv(model_name, X_train, y_target, X_test, all_folds, seed=SEED):
    \"\"\"Single-seed CV for one model.\"\"\"
    n_train = len(X_train)
    n_test  = len(X_test)
    oof_preds  = np.zeros(n_train)
    test_preds = np.zeros(n_test)
    fold_scores = []

    train_func = {
        "LightGBM": train_lgb_fold,
        "XGBoost":  train_xgb_fold,
        "CatBoost": train_cat_fold,
        "HistGBR":  train_hist_fold,
    }[model_name]

    for fold_idx, (tr_idx, va_idx) in enumerate(all_folds, 1):
        X_tr = X_train.iloc[tr_idx]
        y_tr = y_target.iloc[tr_idx]
        X_va = X_train.iloc[va_idx]
        y_va = y_target.iloc[va_idx]

        model = train_func(X_tr, y_tr, X_va, y_va, seed=seed)

        va_pred   = model.predict(X_va)
        test_pred = model.predict(X_test)

        oof_preds[va_idx] = va_pred
        fold_r2 = r2_score(y_va, va_pred)
        fold_scores.append(fold_r2)

        test_preds += test_pred / N_SPLITS

    mean_r2 = float(np.mean(fold_scores))
    return oof_preds, test_preds, mean_r2


def run_model_cv_multiseed(model_name, X_train, y_target, X_test, all_folds,
                           seeds=None):
    \"\"\"3-seed averaging: trains each model with multiple seeds, averages predictions.\"\"\"
    if seeds is None:
        seeds = MULTI_SEEDS
    all_oof  = []
    all_test = []
    for i, seed in enumerate(seeds, 1):
        oof, test_p, r2 = run_model_cv(
            model_name, X_train, y_target, X_test, all_folds, seed=seed
        )
        all_oof.append(oof)
        all_test.append(test_p)
        print(f"  [{model_name}] seed {i}/{len(seeds)} (seed={seed}): CV R\u00b2 = {r2:.5f}")
    avg_oof  = np.mean(all_oof,  axis=0)
    avg_test = np.mean(all_test, axis=0)
    avg_r2   = r2_score(y_target, avg_oof)
    print(f"  [{model_name}] 3-seed avg CV R\u00b2 = {avg_r2:.5f}\\n")
    return avg_oof, avg_test, avg_r2

print("Model functions defined (4 models, 3-seed averaging, GPU configs preserved)")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 12: Pipeline \u2013 Load data, features, lag, KFold, TE, neighbour TE, matrix
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "f574a091",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 PIPELINE: Load Data \u2192 Features \u2192 Lag \u2192 KFold \u2192 TE \u2192 Neighbour TE \u2192 Matrix \u2500

print("Loading data ...")
train_raw = pd.read_csv(TRAIN_PATH)
test_raw  = pd.read_csv(TEST_PATH)

# Raw counts \u2013 Poisson objectives handle the distribution
y = train_raw[TARGET].astype(float)

print("Building base features ...")
train, test = build_base_features(train_raw, test_raw)

# \u2500\u2500 Day-48 Demand Lag (Phase 3) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Building demand lag features ...")
if "geohash" in train.columns and "timestamp" in train.columns:
    # Build lookup from Day 48 training data: (geohash_timestamp) -> demand
    _lag_key_tr = train["geohash"].astype(str) + "_" + train["timestamp"].astype(str)
    _lag_lookup = dict(zip(_lag_key_tr, y))
    _gh_mean_demand = y.groupby(train["geohash"]).mean().to_dict()
    _global_mean_demand = y.mean()

    # Train: no prior day available \u2192 use geohash mean as proxy
    train["demand_lag1"] = train["geohash"].map(_gh_mean_demand).fillna(_global_mean_demand)

    # Test (Day 49): exact match from Day 48, fallback to geohash mean
    _lag_key_te = test["geohash"].astype(str) + "_" + test["timestamp"].astype(str)
    test["demand_lag1"] = _lag_key_te.map(_lag_lookup)
    test["demand_lag1"] = test["demand_lag1"].fillna(
        test["geohash"].map(_gh_mean_demand)
    ).fillna(_global_mean_demand)
    print(f"  demand_lag1: train fill rate = {train['demand_lag1'].notna().mean():.3f}, "
          f"test exact-match rate = {_lag_key_te.map(_lag_lookup).notna().mean():.3f}")

# \u2500\u2500 KFold (10 folds, shuffled) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
kf = KFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
all_folds = list(kf.split(train, y))
print(f"  KFold: {N_SPLITS} folds materialised")

# \u2500\u2500 OOF Target Encoding (mean + std + median) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Applying OOF target encoding (mean/std/median) ...")
train, test = target_encode_oof(train, test, y, all_folds)
print("  Done.")

# \u2500\u2500 Geohash Neighbour Average (Phase 3) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Computing geohash neighbour TE mean ...")
if "geohash" in train.columns and "te_geohash" in train.columns:
    # Build geohash \u2192 te_geohash map from combined train+test
    _all_te = pd.concat([
        train[["geohash", "te_geohash"]],
        test[["geohash", "te_geohash"]]
    ])
    _gh_te_map = _all_te.groupby("geohash")["te_geohash"].mean().to_dict()
    _global_te = np.mean(list(_gh_te_map.values()))

    # Compute neighbour TE for each unique geohash (vectorized lookup)
    _unique_ghs = set(train["geohash"].unique()) | set(test["geohash"].unique())
    _gh_nbr_te = {}
    for _gh in _unique_ghs:
        _te_vals = []
        for _d in ['n', 's', 'e', 'w']:
            _nbr = geohash_neighbour(_gh, _d)
            if _nbr and _nbr in _gh_te_map:
                _te_vals.append(_gh_te_map[_nbr])
        _gh_nbr_te[_gh] = np.mean(_te_vals) if _te_vals else _global_te

    train["neighbor_te_mean"] = train["geohash"].map(_gh_nbr_te).fillna(_global_te)
    test["neighbor_te_mean"]  = test["geohash"].map(_gh_nbr_te).fillna(_global_te)
    print(f"  neighbor_te_mean computed for {len(_gh_nbr_te)} unique geohashes")

# \u2500\u2500 Prepare feature matrix \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
drop_cols = {
    ID_COL, TARGET, "timestamp", "day", "geohash",
    "geohash_prefix_3", "geohash_prefix_4", "geohash_prefix_5",
    "RoadType", "Weather", "cluster_x_hour",
}
features = [c for c in train.columns if c not in drop_cols and c in test.columns]

for col in features:
    train[col] = pd.to_numeric(train[col], errors="coerce")
    test[col]  = pd.to_numeric(test[col],  errors="coerce")

for col in features:
    col_mean   = train[col].mean()
    train[col] = train[col].fillna(col_mean)
    test[col]  = test[col].fillna(col_mean)

X_train = train[features]
X_test  = test[features]
print(f"  {len(features)} features ready")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 13: Train base models (4 models, 3-seed averaging, use y not y_log)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "ee0a0e92",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 SECTION 3: Train base models (3-seed averaging) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Training base models (4 models \u00d7 3 seeds \u00d7 10 folds = 120 fits)...\\n")

oof_lgb,  test_lgb,  r2_lgb  = run_model_cv_multiseed("LightGBM", X_train, y, X_test, all_folds)
oof_xgb,  test_xgb,  r2_xgb  = run_model_cv_multiseed("XGBoost",  X_train, y, X_test, all_folds)
oof_cat,  test_cat,  r2_cat  = run_model_cv_multiseed("CatBoost", X_train, y, X_test, all_folds)
oof_hist, test_hist, r2_hist = run_model_cv_multiseed("HistGBR",  X_train, y, X_test, all_folds)
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 14: Ridge Meta-Learner (replaces scipy minimize)
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "cc9821ac",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 SECTION 4: Ridge Meta-Learner (Stacking) \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print("Fitting Ridge meta-learner ...")

S_train_oof = np.column_stack([oof_lgb, oof_xgb, oof_cat, oof_hist])
S_test_avg  = np.column_stack([test_lgb, test_xgb, test_cat, test_hist])

meta = Ridge(alpha=1.0)
meta.fit(S_train_oof, y)

print(f"  Ridge coefficients: LGB={meta.coef_[0]:.4f}, XGB={meta.coef_[1]:.4f}, "
      f"CAT={meta.coef_[2]:.4f}, Hist={meta.coef_[3]:.4f}")
print(f"  Ridge intercept: {meta.intercept_:.6f}")

final_oof  = meta.predict(S_train_oof)
final_pred = meta.predict(S_test_avg)

blend_cv_r2 = r2_score(y, final_oof)
print(f"  Ridge OOF R\u00b2  = {blend_cv_r2:.5f}")
print(f"  Estimated score = {max(0, 100*blend_cv_r2):.4f}")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 15: Post-processing + Isotonic Calibration
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "2bbfa094",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 SECTION 5: Post-processing + Isotonic Calibration \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500

# 1. Isotonic Calibration: match training target distribution
iso = IsotonicRegression(out_of_bounds='clip')
iso.fit(final_oof, y)
final_pred_calibrated = iso.predict(final_pred)
print("  Isotonic calibration applied")

# 2. Clip to [0, p99.5]
p995 = np.percentile(train_raw[TARGET].dropna().values, 99.5)
final_pred_calibrated = np.clip(final_pred_calibrated, 0, p995)
print(f"  Clipped to [0, {p995:.4f}]")

# 3. Rounding (only if target values are 0.5-step aligned)
vals = train_raw[TARGET].dropna().values
if np.mean(vals % 0.5 == 0) > 0.95:
    final_pred_calibrated = np.round(final_pred_calibrated * 2) / 2
    print("  Applied 0.5-step rounding")
else:
    print("  No rounding applied")

# 4. Save
submission = pd.DataFrame({ID_COL: test_raw[ID_COL], TARGET: final_pred_calibrated})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"  Saved {SUBMISSION_PATH}  shape={submission.shape}")
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# CELL 16: Summary
# ═══════════════════════════════════════════════════════════════════════════════
cells.append({
    "cell_type": "code",
    "execution_count": None,
    "id": "66022963",
    "metadata": {},
    "outputs": [],
    "source": make_source("""
# \u2500\u2500 Summary \u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500\u2500
print()
print("\u2550" * 50)
print(f"  LightGBM  CV R\u00b2 : {r2_lgb:.5f}")
print(f"  XGBoost   CV R\u00b2 : {r2_xgb:.5f}")
print(f"  CatBoost  CV R\u00b2 : {r2_cat:.5f}")
print(f"  HistGBR   CV R\u00b2 : {r2_hist:.5f}")
print(f"  Ridge     OOF R\u00b2: {blend_cv_r2:.5f}")
print(f"  Estimated score : {max(0, 100*blend_cv_r2):.4f}")
print("\u2550" * 50)
""")
})

# ═══════════════════════════════════════════════════════════════════════════════
# Assemble notebook
# ═══════════════════════════════════════════════════════════════════════════════
nb = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "flipkart-env",
            "language": "python",
            "name": "python3"
        },
        "language_info": {
            "codemirror_mode": {"name": "ipython", "version": 3},
            "file_extension": ".py",
            "mimetype": "text/x-python",
            "name": "python",
            "nbformat_exporter": "python",
            "pygments_lexer": "ipython3",
            "version": "3.10.20"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 5
}

with open("updated_nb.ipynb", "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)

print("updated_nb.ipynb generated successfully!")
print(f"  Total cells: {len(cells)}")
