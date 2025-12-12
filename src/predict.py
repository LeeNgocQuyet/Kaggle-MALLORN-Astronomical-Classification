# src/predict.py
import pandas as pd
import joblib
import os
import numpy as np

MODEL_DIR = "models"
TEST_FINAL = "data/processed/test_final.csv"
OUT = "submissions/svm_submission.csv"

def run():
    # load artifacts
    clf = joblib.load(os.path.join(MODEL_DIR,"svm_model.pkl"))
    scaler = joblib.load(os.path.join(MODEL_DIR,"svm_scaler.pkl"))
    feature_cols = joblib.load(os.path.join(MODEL_DIR,"svm_features.pkl"))
    medians = joblib.load(os.path.join(MODEL_DIR,"svm_medians.pkl"))

    df_test = pd.read_csv(TEST_FINAL)
    df_test['object_id'] = df_test['object_id'].astype(str).str.strip()

    # ensure all feature_cols present; if missing, create and fill with median 0
    for c in feature_cols:
        if c not in df_test.columns:
            df_test[c] = np.nan

    X = df_test[feature_cols].fillna(medians)
    X_scaled = scaler.transform(X)

    preds = clf.predict(X_scaled)
    probs = clf.predict_proba(X_scaled)[:,1]

    sub = pd.DataFrame({
        "object_id": df_test["object_id"],
        "predict": preds,       # discrete label
    })
    os.makedirs("submissions", exist_ok=True)
    sub.to_csv(OUT, index=False)
    print("Saved submission to", OUT)

if __name__ == "__main__":
    run()
