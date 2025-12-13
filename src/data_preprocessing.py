# src/data_preprocessing.py
import pandas as pd
import os
from glob import glob

RAW_ROOT = "data/raw"
OUT_DIR = "data/processed"


# ---------- UTILS ----------
def normalize_cols(df):
    rename_map = {}
    for c in df.columns:
        k = c.lower().replace(" ", "").replace("_", "")
        if k in ["time(mjd)", "timemjd", "time"]:
            rename_map[c] = "mjd"
        elif k == "flux":
            rename_map[c] = "flux"
        elif k in ["fluxerr", "flux_err"]:
            rename_map[c] = "flux_err"
        elif k in ["filter", "passband"]:
            rename_map[c] = "filter"
        elif k in ["objectid", "object_id"]:
            rename_map[c] = "object_id"
    if rename_map:
        df = df.rename(columns=rename_map)
    return df


# ---------- FEATURE EXTRACTION ----------
def extract_features(df):
    df = normalize_cols(df)

    required = {"object_id", "mjd", "flux", "flux_err"}
    if not required.issubset(df.columns):
        raise ValueError(f"Missing columns: {required - set(df.columns)}")

    df["object_id"] = df["object_id"].astype(str).str.strip()

    g = df.groupby("object_id")

    feats = g.agg(
        mean_flux=("flux", "mean"),
        std_flux=("flux", "std"),
        min_flux=("flux", "min"),
        max_flux=("flux", "max"),
        mean_flux_err=("flux_err", "mean"),
        num_points=("flux", "count"),
        duration=("mjd", lambda x: x.max() - x.min()),
    ).reset_index()

    return feats


# ---------- PROCESS ONE SPLIT ----------
def process_split(split_name):
    split_dir = os.path.join(RAW_ROOT, split_name)

    paths = {
        "train": os.path.join(split_dir, "train_full_lightcurves.csv"),
        "test":  os.path.join(split_dir, "test_full_lightcurves.csv"),
    }

    outputs = {}

    for kind, path in paths.items():
        if not os.path.exists(path):
            continue

        df = pd.read_csv(path)
        feats = extract_features(df)

        out = os.path.join(OUT_DIR, f"{kind}_features_{split_name}.csv")
        feats.to_csv(out, index=False)
        outputs[kind] = out

        print(f"✅ {split_name} | {kind}: {len(feats)} objects")

    return outputs


# ---------- COMBINE SPLITS ----------
def combine(kind):
    parts = sorted(glob(os.path.join(OUT_DIR, f"{kind}_features_split_*.csv")))
    if not parts:
        raise FileNotFoundError(f"No {kind} feature files found")

    df = pd.concat([pd.read_csv(p) for p in parts], ignore_index=True)
    df["object_id"] = df["object_id"].astype(str).str.strip()

    if len(df) != df["object_id"].nunique():
        raise ValueError(f"❌ DUPLICATE object_id in combined {kind}")

    out = os.path.join(OUT_DIR, f"{kind}_features_all.csv")
    df.to_csv(out, index=False)

    print(f"🎯 Combined {kind}: {len(df)} rows → {out}")
    return out


# ---------- MAIN ----------
def run():
    os.makedirs(OUT_DIR, exist_ok=True)

    splits = [f"split_{i:02d}" for i in range(1, 21)]

    print("🚀 Processing all splits...")
    for s in splits:
        process_split(s)

    print("🔗 Combining splits...")
    combine("train")
    combine("test")

    print("✅ data_preprocessing COMPLETED (ALL DATA, NO DUPLICATES)")


if __name__ == "__main__":
    # tự động lấy tất cả split_01 → split_20
    splits = [f"split_{i:02d}" for i in range(1, 21)]
    run(splits)