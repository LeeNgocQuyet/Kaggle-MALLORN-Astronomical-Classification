# src/train_svm.py
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import roc_auc_score, accuracy_score

TRAIN_FINAL = "data/processed/train_final.csv"
MODEL_DIR = "models"

def load_data(path=TRAIN_FINAL):
    return pd.read_csv(path)

def get_numeric_feature_cols(df):
    # choose all numeric columns except object_id and target
    exclude = {'object_id','target'}
    # pick columns with numeric dtypes or convertible to numeric
    cols = []
    for c in df.columns:
        if c in exclude:
            continue
        # try convert
        try:
            pd.to_numeric(df[c], errors='raise')
            cols.append(c)
        except Exception:
            # non-numeric, skip
            pass
    return cols

def preprocess(df, feature_cols):
    # fill NaN with median of column
    medians = df[feature_cols].median()
    X = df[feature_cols].fillna(medians)
    y = df['target'].astype(int)
    return X, y, medians

def run(cv_folds=5):
    df = load_data()
    print("Loaded train_final:", df.shape)

    feature_cols = get_numeric_feature_cols(df)
    print("Feature candidates:", feature_cols)

    # remove accidental small features (ensure at least 2 features)
    if 'target' in feature_cols:
        feature_cols.remove('target')
    if 'object_id' in feature_cols:
        feature_cols.remove('object_id')

    X_df, y, medians = preprocess(df, feature_cols)

    print("Class distribution:", y.value_counts().to_dict())

    # Scale
    scaler = StandardScaler()
    X = scaler.fit_transform(X_df)

    # Cross-validation
    clf = SVC(kernel="rbf", C=3, gamma="scale", class_weight="balanced", probability=True, random_state=42)
    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
    aucs = []
    accs = []
    for train_idx, val_idx in skf.split(X, y):
        clf.fit(X[train_idx], y.iloc[train_idx])
        preds = clf.predict(X[val_idx])
        probs = clf.predict_proba(X[val_idx])[:,1]
        accs.append(accuracy_score(y.iloc[val_idx], preds))
        try:
            aucs.append(roc_auc_score(y.iloc[val_idx], probs))
        except Exception:
            aucs.append(np.nan)
    print(f"CV accuracy: {np.mean(accs):.4f} ± {np.std(accs):.4f}")
    print(f"CV AUC: {np.nanmean(aucs):.4f} ± {np.nanstd(aucs):.4f}")

    # Train final on full data
    clf.fit(X, y)

    # save artifacts
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODEL_DIR,"svm_model.pkl"))
    joblib.dump(scaler, os.path.join(MODEL_DIR,"svm_scaler.pkl"))
    joblib.dump(feature_cols, os.path.join(MODEL_DIR,"svm_features.pkl"))
    joblib.dump(medians, os.path.join(MODEL_DIR,"svm_medians.pkl"))

    print("Saved model + artifacts to", MODEL_DIR)

if __name__ == "__main__":
    run()
