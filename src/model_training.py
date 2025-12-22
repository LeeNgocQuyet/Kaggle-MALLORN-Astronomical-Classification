"""
End-to-end training script for baseline SVM pipeline.
Usage example:
  python src/model_training.py --base-path data/raw --out submission.csv
"""
import argparse
import os
import pandas as pd
import numpy as np
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import f1_score, classification_report
from imblearn.over_sampling import SMOTE
from src.data_processing import load_all_splits
from src.utils import save_model, save_submission
import warnings
warnings.filterwarnings('ignore')


def run_pipeline(base_path, out_path):
    # 1. load features
    print('Loading train features...')
    train_lc_features = load_all_splits(base_path, mode='train')
    print('Loading test features...')
    test_lc_features = load_all_splits(base_path, mode='test')

    # 2. load logs
    train_log = pd.read_csv(os.path.join(os.path.dirname(base_path), 'train_log.csv'))
    test_log = pd.read_csv(os.path.join(os.path.dirname(base_path), 'test_log.csv'))

    # 3. merge
    full_train = train_log.merge(train_lc_features, on='object_id', how='left')
    full_test = test_log.merge(test_lc_features, on='object_id', how='left')
    full_train.fillna(0, inplace=True)
    full_test.fillna(0, inplace=True)

    drop_cols = ['object_id', 'SpecType', 'English Translation', 'split', 'target', 'Z_err']
    feature_cols = [c for c in full_train.columns if c not in drop_cols]

    X = full_train[feature_cols]
    y = full_train['target']
    X_test_final = full_test[feature_cols]

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_test_scaled = scaler.transform(X_test_final)

    print('Applying SMOTE...')
    smote = SMOTE(random_state=42)
    X_resampled, y_resampled = smote.fit_resample(X_scaled, y)

    X_train, X_val, y_train, y_val = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

    svm = SVC(kernel='rbf', probability=True, random_state=42)
    param_grid = {'C': [1, 10, 100], 'gamma': ['scale', 0.1, 0.01]}
    grid = GridSearchCV(svm, param_grid, cv=3, scoring='f1', verbose=2, n_jobs=-1)
    grid.fit(X_train, y_train)
    best_model = grid.best_estimator_

    y_pred_val = best_model.predict(X_val)
    print('Validation F1:', f1_score(y_val, y_pred_val))
    print(classification_report(y_val, y_pred_val))

    # final predict
    final_predictions = best_model.predict(X_test_scaled)
    submission = pd.DataFrame({'object_id': full_test['object_id'], 'prediction': final_predictions})
    save_submission(submission, out_path)
    print('Saved submission to', out_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--base-path', type=str, default='data/raw', help='Base path to split_* folders')
    parser.add_argument('--out', type=str, default='submission_svm_improved.csv')
    args = parser.parse_args()
    run_pipeline(args.base_path, args.out)
