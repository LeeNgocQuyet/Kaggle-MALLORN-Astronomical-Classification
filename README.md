Kaggle-MALLORN-Astronomical-Classification

This repository implements a complete machine learning pipeline for the
Kaggle MALLORN Astronomical Classification Challenge, with the goal of
detecting Tidal Disruption Events (TDEs) from astronomical lightcurves.

1. Problem Overview

Task: Binary classification

# Kaggle MALLORN — Astronomical Classification

Một pipeline ML đơn giản cho bài toán phân loại Tidal Disruption Events (TDE) trên Kaggle.

## Tổng quan
- Bài toán: phân loại nhị phân (0 = Non-TDE, 1 = TDE)
- Dữ liệu: lightcurves từ nhiều split (split_01 ... split_20)
- Mô hình baseline: SVM

## Bắt đầu nhanh (Quickstart)
1. Tạo môi trường Python và cài phụ thuộc:

```bash
python -m venv .venv
# Windows (PowerShell)
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

2. Chạy pipeline (các script nằm trong `src/`):

```bash
# 1) Trích xuất đặc trưng
python src/data_preprocessing.py

# 2) Gộp features với nhãn
python src/build_train_final.py

# 3) Huấn luyện SVM
python src/train_svm.py

# 4) Chuẩn hoá test và tạo bảng cuối
python src/preprocess_test.py

# 5) Dự đoán và tạo file nộp
python src/svm_predict.py
```

Output tiêu biểu:
- `data/processed/train_final.csv`, `data/processed/test_final.csv`
- `models/svm_model.pkl`, `models/svm_scaler.pkl`
- `submissions/svm_submission.csv`

## Cấu trúc repository

- `data/raw/` — (không lưu trong repo) chứa dữ liệu gốc theo từng `split_XX/`
- `data/processed/` — bảng features và files đã xử lý
- `src/` — script pipeline và xử lý
- `models/` — artifacts huấn luyện
- `notebooks/` — notebook khám phá dữ liệu

## Ghi chú
- Không lưu dữ liệu raw trong repo; đặt file Kaggle dưới `data/raw/` theo cấu trúc split.
- Có `CODE_OF_CONDUCT.md` để hướng dẫn đóng góp.
- License: không có license trong repo theo yêu cầu.

## Muốn đóng góp?
- Mở issue hoặc PR; tuân thủ `CODE_OF_CONDUCT.md`.

## Tác giả
Quyết Lê Ngọc
