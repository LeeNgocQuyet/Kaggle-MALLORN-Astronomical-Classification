import pandas as pd
import joblib
import os
import numpy as np

MODEL_DIR = "models"
TEST_FINAL = "data/processed/test_final.csv"
OUT = "submissions/svm_submission.csv"

def run():
    clf = joblib.load(os.path.join(MODEL_DIR,"svm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR,"svm_scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR,"svm_features.pkl"))
    medians = joblib.load(os.path.join(MODEL_DIR,"svm_medians.pkl"))

    df = pd.read_csv(TEST_FINAL)
    df["object_id"] = df["object_id"].astype(str).str.strip()

    # ensure all features exist
    for c in feature_cols:
        if c not in df.columns:
            df[c] = np.nan

    X = df[feature_cols].fillna(medians)
    X_scaled = scaler.transform(X)

    preds = clf.predict(X_scaled)

    sub = pd.DataFrame({
        "object_id": df["object_id"],
        "prediction": preds
    })

    os.makedirs("submissions", exist_ok=True)
    sub.to_csv(OUT, index=False)

    print("Saved:", OUT)
    print("Rows:", len(sub), "Unique:", sub.object_id.nunique())
    print(sub.prediction.value_counts())

if __name__ == "__main__":
    run()
