# MALLORN Astronomical Classification

## Overview
This project targets the **MALLORN Astronomical Classification Challenge**, classifying astronomical objects (TDE vs. Non-TDE) based on their lightcurve data.

The solution implements a robust Machine Learning pipeline featuring:
- **Feature Engineering**: Aggregated statistics and colors extracted from raw lightcurves.
- **Data Augmentation**: SMOTE to handle class imbalance (TDEs are rare).
- **Model Tuning**: SVM with RBF kernel and XGBoost, optimized via Grid/Randomized Search.
- **Refinement**: Probability Calibration and Threshold Tuning to maximize F1-score.

## Methodology

### 1. Feature Extraction
We process raw lightcurves (Flux vs Time) for 6 filters (u, g, r, i, z, y) to extract:
- **Statistics**: Max, Min, Mean, Std, Skewness, SNR.
- **Colors**: Difference in Flux between bands (e.g., `g - r`).
- **Amplitude**: Variability range per filter.

### 2. Preprocessing
- **Imputation**: Median filling for missing values.
- **Scaling**: `StandardScaler` (for SVM) and `RobustScaler` (for XGBoost).
- **Imbalance Handling**: `SMOTE` (Synthetic Minority Over-sampling Technique) generates synthetic TDE samples during training to improve recall.

### 3. Models
We focused on two primary architectures:
1.  **SVM (Support Vector Machine)**:
    -   Kernel: RBF (captures non-linearities).
    -   Optimization: Tuned `C` and `gamma`.
    -   **Calibration**: Applied Isotonic Calibration to refine probability estimates.
2.  **XGBoost (Gradient Boosting)**:
    -   Tree-based ensemble for feature importance analysis.

## Results

**Best Model**: SVM (Calibrated + Threshold Tuned)
- **Validation F1-Score**: **0.4590**
- **Optimal Threshold**: **0.375**

*Note: The F1 score reflects the challenging nature of the dataset and class imbalance. Calibration significantly improved reliability.*

## Repository Structure

```
├── data/
│   ├── raw/               # Raw splits and logs
│   └── processed/         # (Optional) Cached features
├── notebooks/
│   ├── 01_svm_classification.ipynb      # Canonical SVM Pipeline
│   └── 02_xgboost_classification.ipynb  # XGBoost Experiment
├── src/
│   ├── data_processing.py # Feature extraction logic
│   ├── train.py           # Training script
│   ├── predict.py         # Inference script
│   └── utils.py           # Helpers
├── experiments/           # Training artifacts (models, manifests)
└── svm_submission.csv     # Final submission file
```

## Reproducibility

### 1. Environment
```bash
pip install -r requirements.txt
```

### 2. Training
To retrain the best model (SVM):
```bash
python -m src.train --base-path data/raw --exp exp08_final
```
*Outputs model to `models/exp08_final_model.joblib` and metrics to `experiments/exp08_final/manifest.json`.*

### 3. Inference
To generate a submission using the trained model:
```bash
python -m src.predict --base-path data/raw --manifest experiments/exp08_final/manifest.json --out svm_submission.csv
```

## Authors
- **[Your Name/Team]**
