# src/data_preprocessing.py
import pandas as pd
import numpy as np
import os
from glob import glob

RAW_ROOT = "data/raw"
OUT_DIR = "data/processed"

def normalize_cols(df):
    # chuẩn hoá tên cột (handle "Time (MJD)" / "Flux" / "Flux_err" / "Filter")
    rename_map = {}
    for c in df.columns:
        if c.lower().replace(" ", "") in ["time(mjd)", "time(mjd)".lower(), "time"]:
            rename_map[c] = "mjd"
        if c.lower().replace(" ", "") in ["flux"]:
            rename_map[c] = "flux"
        if c.lower().replace(" ", "") in ["flux_err", "fluxerr"]:
            rename_map[c] = "flux_err"
        if c.lower().replace(" ", "") in ["filter", "passband"]:
            rename_map[c] = "filter"
        if c.lower().replace(" ", "") in ["objectid", "object_id", "objectid"]:
            rename_map[c] = "object_id"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df

def extract_features_from_df(df):
    df = normalize_cols(df)
    # required cols check
    needed = {"object_id", "mjd", "flux", "flux_err"}
    if not needed.issubset(set(df.columns)):
        missing = needed - set(df.columns)
        raise KeyError(f"Missing required columns in LC: {missing}")
    g = df.groupby("object_id")
    feats = g.agg(
        mean_flux = ("flux", "mean"),
        std_flux  = ("flux", "std"),
        min_flux  = ("flux", "min"),
        max_flux  = ("flux", "max"),
        mean_flux_err = ("flux_err", "mean"),
        num_points = ("flux", "count"),
        duration = ("mjd", lambda x: x.max() - x.min())
    ).reset_index()
    return feats

def process_split(split_name):
    split_dir = os.path.join(RAW_ROOT, split_name)
    train_path = os.path.join(split_dir, "train_full_lightcurves.csv")
    test_path  = os.path.join(split_dir, "test_full_lightcurves.csv")
    out_train = os.path.join(OUT_DIR, f"train_features_{split_name}.csv")
    out_test  = os.path.join(OUT_DIR, f"test_features_{split_name}.csv")

    print(f"Processing {split_name}:")
    if os.path.exists(train_path):
        df_train = pd.read_csv(train_path)
        df_train = normalize_cols(df_train)
        feats_train = extract_features_from_df(df_train)
        feats_train.to_csv(out_train, index=False)
        print(f"  saved {out_train} ({len(feats_train)} objects)")
    else:
        print(f"  no train file at {train_path}")

    if os.path.exists(test_path):
        df_test = pd.read_csv(test_path)
        df_test = normalize_cols(df_test)
        feats_test = extract_features_from_df(df_test)
        feats_test.to_csv(out_test, index=False)
        print(f"  saved {out_test} ({len(feats_test)} objects)")
    else:
        print(f"  no test file at {test_path}")

def combine_splits(splits, kind="train"):
    # combine train_features_split_xxx.csv into single file
    os.makedirs(OUT_DIR, exist_ok=True)
    parts = []
    for s in splits:
        p = os.path.join(OUT_DIR, f"{kind}_features_{s}.csv")
        if os.path.exists(p):
            parts.append(p)
    if not parts:
        print(f"No {kind} parts found for splits: {splits}")
        return None
    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    out = os.path.join(OUT_DIR, f"{kind}_features_all.csv")
    df.to_csv(out, index=False)
    print(f"Saved combined {kind} features to {out} ({len(df)} rows)")
    return out

def run(splits=None):
    if splits is None:
        splits = ["split_01"]  # default fast run
    os.makedirs(OUT_DIR, exist_ok=True)
    for s in splits:
        process_split(s)
    # combine across splits
    train_comb = combine_splits(splits, kind="train")
    test_comb  = combine_splits(splits, kind="test")
    return train_comb, test_comb

if __name__ == "__main__":
    # Example: default will process split_01 only
    run(["split_01"])
    # To process all -> run(["split_01","split_02",...,"split_20"])
