"""Train script for MALLORN experiments.

Usage example:
  python src/train.py --base-path data/raw --exp exp07 --model-dir models --out-dir experiments/exp07
"""
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
from sklearn.metrics import f1_score
from imblearn.pipeline import Pipeline as ImbPipeline
from imblearn.over_sampling import SMOTE

from src.data_processing import load_all_splits


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', required=True, help='Base path to dataset (contains split_01..split_20 and train_log.csv)')
    parser.add_argument('--exp', required=True, help='Experiment id (used to name artifacts)')
    parser.add_argument('--model-dir', default='models', help='Directory to save model')
    parser.add_argument('--out-dir', default=None, help='Experiment output directory (manifest, metrics)')
    parser.add_argument('--random-state', type=int, default=42)
    args = parser.parse_args(argv)

    base = args.base_path
    exp = args.exp
    model_dir = args.model_dir
    out_dir = args.out_dir or os.path.join('experiments', exp)

    os.makedirs(model_dir, exist_ok=True)
    os.makedirs(out_dir, exist_ok=True)

    # 1) Load features
    print('Loading train features...')
    train_feats = load_all_splits(base, mode='train')
    if train_feats.empty:
        raise SystemExit(f'No train features found under {base}')

    train_log_path = os.path.join(base, 'train_log.csv')
    train_log = pd.read_csv(train_log_path)

    full_train = train_log.merge(train_feats, on='object_id', how='left')
    full_train.fillna(0, inplace=True)

    drop_cols = ['object_id', 'SpecType', 'English Translation', 'split', 'target', 'Z_err']
    feature_cols = [c for c in full_train.columns if c not in drop_cols]

    X = full_train[feature_cols]
    y = full_train['target']

    # split train/val
    X_train_org, X_val_org, y_train_org, y_val_org = train_test_split(
        X, y, test_size=0.2, random_state=args.random_state, stratify=y
    )

    # 2) Build pipeline
    pipeline = ImbPipeline([
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler()),
        ('smote', SMOTE(random_state=args.random_state)),
        ('select', SelectKBest(score_func=f_classif)),
        ('svm', SVC(probability=True, random_state=args.random_state))
    ])

    param_grid = {
        'select__k': [20, 30],
        'svm__C': [10, 50],
        'svm__gamma': ['scale', 0.1],
        'smote__sampling_strategy': [0.5, 0.75]
    }

    print('Starting GridSearchCV...')
    grid = GridSearchCV(pipeline, param_grid, cv=3, scoring='f1', verbose=1, n_jobs=-1)
    grid.fit(X_train_org, y_train_org)

    best = grid.best_estimator_
    print('Best params:', grid.best_params_)

    # evaluate on val
    y_val_pred = best.predict(X_val_org)
    val_f1 = f1_score(y_val_org, y_val_pred)
    print('Validation F1:', val_f1)

    # save model
    model_path = os.path.join(model_dir, f'{exp}_model.joblib')
    joblib.dump(best, model_path)

    # write manifest
    manifest = {
        'id': exp,
        'date': datetime.utcnow().isoformat() + 'Z',
        'best_params': grid.best_params_,
        'val_f1': float(val_f1),
        'model_path': model_path
    }
    manifest_path = os.path.join(out_dir, 'manifest.json')
    with open(manifest_path, 'w') as f:
        json.dump(manifest, f, indent=2)

    print('Saved model to', model_path)
    print('Wrote manifest to', manifest_path)


if __name__ == '__main__':
    main()
