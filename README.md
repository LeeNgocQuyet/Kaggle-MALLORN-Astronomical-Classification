Kaggle-MALLORN-Astronomical-Classification

This repository implements a complete machine learning pipeline for the
Kaggle MALLORN Astronomical Classification Challenge, with the goal of
detecting Tidal Disruption Events (TDEs) from astronomical lightcurves.

1. Problem Overview

Task: Binary classification

0: Non-TDE

1: TDE

Data: Astronomical lightcurves collected over ~10 years

Main challenges:

Strong class imbalance

Irregular time series

Baseline model: Support Vector Machine (SVM)

2. Repository Structure
Kaggle-MALLORN-Astronomical-Classification/
│
├── data/
│   ├── raw/                 # Kaggle original data (NOT tracked)
│   └── processed/           # Generated feature tables
│
├── models/                  # Trained model artifacts
│
├── submissions/             # Kaggle submission files
│
├── src/
│   ├── data_preprocessing.py    # Feature extraction from all splits
│   ├── build_train_final.py     # Merge features + labels
│   ├── train_svm.py             # Train SVM model
│   ├── preprocess_test.py       # Build test_final.csv
│   └── svm_predict.py           # Generate submission
│
├── notebooks/
│   └── eda.ipynb                # Optional EDA
│
├── requirements.txt
├── .gitignore
└── README.md

3. Installation

Create a Python environment (recommended) and install dependencies:

pip install -r requirements.txt

4. End-to-End Pipeline
Step 1: Feature Extraction (All Splits)

Extract lightcurve-level features from raw CSV files:

python src/data_preprocessing.py


Generated files:

data/processed/train_features_all.csv

data/processed/test_features_all.csv

Step 2: Build Training Table

Merge extracted features with training labels:

python src/build_train_final.py


Output:

data/processed/train_final.csv

Step 3: Train SVM Model

Train an SVM with class balancing and cross-validation:

python src/train_svm.py


Saved artifacts:

models/svm_model.pkl

models/svm_scaler.pkl

models/svm_features.pkl

models/svm_medians.pkl

Step 4: Preprocess Test Data

Build final test table used for inference:

python src/preprocess_test.py


Output:

data/processed/test_final.csv

Step 5: Generate Kaggle Submission

Generate submission file:

python src/svm_predict.py


Output:

submissions/svm_submission.csv

Submission format:

object_id,prediction

5. Current Results

Model: SVM (RBF kernel, class_weight=balanced)

Public Kaggle score: ~0.13

This score serves as a baseline.

6. Notes on Reproducibility

Raw Kaggle data is not included in this repository.

Place Kaggle files under data/raw/ following the competition structure.

The pipeline supports running from split_01 up to split_20.

7. Planned Improvements

Per-filter (u, g, r, i, z, y) feature extraction

Time-domain features (slopes, variability, skewness)

Gradient boosting models (XGBoost / LightGBM)

Threshold optimization for imbalanced classification

8. Author

Quyết Lê Ngọc

This repository is part of a machine learning project for astronomical
event classification and Kaggle competition participation.