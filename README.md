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
# MALLORN — Phân loại thiên văn

## Tổng quan
Kho này chứa mã nguồn và ghi chép cho cuộc thi MALLORN Astronomical Classification (phân loại đối tượng thiên văn: TDE vs Non-TDE) dựa trên dữ liệu đường cong sáng (lightcurve).

Mục tiêu: có một pipeline tái lập được (reproducible) gồm trích xuất đặc trưng, huấn luyện mô hình, tối ưu và sinh file nộp bài (`svm_submission.csv`).

## Tóm tắt phương pháp

- Trích xuất đặc trưng từ lightcurve theo từng filter (u,g,r,i,z,y): thống kê (max, min, mean, std), độ lệch (skew), SNR, các màu (color) như `g - r`, và số lượng quan sát.
- Tiền xử lý: điền thiếu bằng median, scaling tùy mô hình (`StandardScaler` cho SVM), và xử lý mất cân bằng bằng SMOTE khi huấn luyện.
- Mô hình chính: SVM (RBF, được calibration xác suất và tuning ngưỡng) và XGBoost để đối chiếu và phân tích quan trọng đặc trưng.

## Kết quả (ví dụ)

- Mô hình tốt nhất (thử nghiệm): SVM đã hiệu chỉnh — Validation F1 ≈ 0.459, ngưỡng tốt nhất ≈ 0.375. (Số này là kết quả thí dụ; xem `experiments/` để có manifest thực tế.)

## Cấu trúc repository

```
data/                 # raw/ và processed/ (dữ liệu lớn không commit vào git)
notebooks/            # Notebook thí nghiệm và pipeline minh họa
src/                  # Mã nguồn: data_processing.py, train.py, predict.py, utils.py
experiments/          # Kết quả huấn luyện, manifest.json, đồ thị
models/               # Model lưu sẵn (.joblib)
requirements.txt      # Phụ thuộc Python
README.md             # Tài liệu (bạn đang xem)
```

## Hướng dẫn nhanh (reproducible)

1) Cài môi trường:

```bash
pip install -r requirements.txt
```

2) Huấn luyện (ví dụ):

```bash
python -m src.train --base-path data/raw --exp exp08_final
```

Model và số liệu sẽ được lưu vào `models/` và `experiments/exp08_final/manifest.json`.

3) Sinh file nộp bài:

```bash
python -m src.predict --base-path data/raw --manifest experiments/exp08_final/manifest.json --out svm_submission.csv
```

## Kiểm thử

- Chạy `pytest` để kiểm tra các hàm xử lý dữ liệu và pipeline.

## Ghi chú cho người chấm

- Trong báo cáo, trình bày rõ: feature engineering, lựa chọn mô hình và lý do, tuning siêu tham số, các metric dùng để đánh giá (F1, Precision, Recall), và các nỗ lực cải tiến (calibration, threshold tuning, augmentation).

## Tác giả

- [Tên bạn / Nhóm]
