"""Train script matching User's SVM Notebook logic."""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import f1_score, precision_recall_curve, classification_report
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_processing import load_all_splits

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', required=True)
    parser.add_argument('--exp', required=True)
    parser.add_argument('--model-dir', default='models')
    parser.add_argument('--out-dir', default=None)
    parser.add_argument('--random-state', type=int, default=42)
    parser.add_argument('--n-jobs', type=int, default=-1)
    args = parser.parse_args(argv)

    base = args.base_path
    exp = args.exp
    model_dir = args.model_dir
    out_dir = args.out_dir or os.path.join('experiments', exp)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1. Load Data
    print('Loading train features...')
    train_feats = load_all_splits(base, mode='train')
    train_log = pd.read_csv(os.path.join(base, 'train_log.csv'))
    
    full_train = train_log.merge(train_feats, on='object_id', how='left')
    full_train.fillna(0, inplace=True)

    drop_cols = ['object_id', 'SpecType', 'English Translation', 'split', 'target', 'Z_err']
    feature_cols = [c for c in full_train.columns if c not in drop_cols]
    
    print(f"Using {len(feature_cols)} features.")

    X = full_train[feature_cols]
    y = full_train['target']

    # 2. Split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    # 3. Pipeline
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=args.random_state)),
        ('svm', SVC(kernel='rbf', probability=True, random_state=args.random_state))
    ])

    # 4. Grid Search
    # User's grid
    param_grid = {
        'svm__C': [1, 10, 100],
        'svm__gamma': ['scale', 0.1, 0.01]
    }

    print('Starting GridSearchCV...')
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=args.n_jobs)
    grid.fit(X_train, y_train)

    best_model = grid.best_estimator_
    print('Best params:', grid.best_params_)

    # 5. Threshold Tuning
    y_val_prob = best_model.predict_proba(X_val)[:, 1]
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_val_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-9)
    best_idx = np.argmax(f1_scores)
    best_threshold = float(thresholds[best_idx])
    best_f1 = float(f1_scores[best_idx])

    print(f"Best Threshold: {best_threshold:.4f}")
    print(f"Max Validation F1: {best_f1:.4f}")

    # Eval
    y_pred_tuned = (y_val_prob >= best_threshold).astype(int)
    print(classification_report(y_val, y_pred_tuned))

    # 6. Save
    model_name = f'{exp}_model.joblib'
    model_path = os.path.join(model_dir, model_name)
    joblib.dump(best_model, model_path)

    manifest = {
        'id': exp,
        'date': datetime.utcnow().isoformat() + 'Z',
        'best_params': grid.best_params_,
        'best_threshold': best_threshold,
        'val_f1': best_f1,
        'model_path': model_path,
        'feature_cols': feature_cols
    }
    
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print('Saved model to', model_path)

if __name__ == '__main__':
    main()
