# Online Transaction Fraud Detection

## End-to-End Machine Learning & Deep Learning Pipeline

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-Latest-green.svg)](https://xgboost.readthedocs.io/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)

---

## Table of Contents
- [Project Overview](#-project-overview)
- [Problem Statement](#-problem-statement)
- [Dataset Information](#-dataset-information)
- [Methodology](#-methodology)
- [Models Implemented](#-models-implemented)
- [Results & Performance](#-results--performance)
- [Key Insights](#-key-insights)
- [Conclusion](#-conclusion)
- [How to Run](#-how-to-run)
- [File Structure](#-file-structure)

---

## Project Overview

Proyek ini membangun pipeline **Machine Learning** dan **Deep Learning** yang komprehensif untuk mendeteksi transaksi fraud dalam data transaksi online. Pipeline mencakup:

1. Exploratory Data Analysis (EDA)
2. Data Preprocessing & Cleaning
3. Handling Class Imbalance
4. Training Traditional ML Models
5. Building Deep Learning Model
6. Model Comparison & Selection
7. Test Set Prediction

---

## Problem Statement

**Online transaction fraud** merupakan masalah kritis dalam industri keuangan yang dapat menyebabkan:
- Kerugian finansial bagi pelanggan dan bisnis
- Hilangnya kepercayaan pelanggan
- Sanksi regulasi

### Tantangan utama dalam fraud detection:

| Challenge | Description |
|-----------|-------------|
| **Class Imbalance** | Transaksi fraud sangat jarang (<5% dari total transaksi) |
| **Evolving Patterns** | Penipu terus mengubah taktik mereka |
| **Real-time Detection** | Kebutuhan prediksi yang cepat dan akurat |

### Target Variable:
- `isFraud = 1`: Transaksi Fraudulent
- `isFraud = 0`: Transaksi Legitimate

---

## Dataset Information

### Dataset Size

| Dataset | Transactions | Features |
|---------|--------------|----------|
| **Training** | 590,540 | 393 + target |
| **Test** | 506,691 | 393 |

### Class Distribution (Highly Imbalanced)

| Class | Count | Percentage |
|-------|-------|------------|
| Not Fraud (0) | ~570,000 | ~96.5% |
| Fraud (1) | ~20,000 | ~3.5% |

**Imbalance Ratio: 1:27** - Setiap 1 transaksi fraud, ada 27 transaksi legitimate.

### Feature Types
- **Categorical Features**: 6 columns (ProductCD, card4, card6, P_emaildomain, R_emaildomain, M4)
- **Numerical Features**: 386+ columns (TransactionAmt, V1-V339, C1-C14, D1-D15, etc.)

---

## Methodology

### 1. Data Loading
- Menggunakan **Polars** untuk data manipulation yang lebih cepat dan memory-efficient

### 2. Exploratory Data Analysis (EDA)
- Analisis distribusi target variable
- Identifikasi missing values
- Analisis tipe fitur (categorical vs numerical)

### 3. Data Preprocessing

| Step | Method | Details |
|------|--------|---------|
| **Missing Values (Numerical)** | Median Imputation | Robust terhadap outliers |
| **Missing Values (Categorical)** | Fill with 'Unknown' | Mempertahankan informasi |
| **Encoding** | Label Encoding | Mengubah kategori ke numerik |
| **Scaling** | StandardScaler | Mean=0, Std=1 |

### 4. Train-Validation Split
- **Training Set**: 80% (472,432 samples)
- **Validation Set**: 20% (118,108 samples)
- **Stratified Split**: Menjaga distribusi fraud ratio

### 5. Handling Class Imbalance

**Approach: Class Weights** (Memory-Efficient Alternative to SMOTE)

```
Class Weight (Not Fraud): ~0.52
Class Weight (Fraud): ~14.30
```

Fraud class diberi bobot **~27x lebih tinggi** untuk menyeimbangkan learning.

**Mengapa Class Weights bukan SMOTE?**
- Memory efficient untuk dataset besar
- Tidak membuat synthetic data
- Lebih cepat training time
- Menghindari overfitting pada synthetic samples

---

## Models Implemented

### 1. Logistic Regression (Baseline)
```python
LogisticRegression(
    C=0.1,
    solver='lbfgs',
    max_iter=1000,
    class_weight='balanced'
)
```
- Linear model sebagai baseline
- Fast training dan interpretable

### 2. Random Forest (Ensemble)
```python
RandomForestClassifier(
    n_estimators=100,
    max_depth=15,
    class_weight='balanced'
)
```
- Ensemble dari decision trees
- Handles non-linear relationships
- Feature importance available

### 3. XGBoost (Gradient Boosting)
```python
XGBClassifier(
    n_estimators=150,
    max_depth=6,
    learning_rate=0.05,
    scale_pos_weight=27  # Handle imbalance
)
```
- State-of-the-art untuk tabular data
- Built-in handling untuk class imbalance
- Excellent performance dan speed

### 4. Neural Network - MLP (Deep Learning)
```
Architecture:
Input (393) → Dense(256) → BN → Dropout(0.3)
           → Dense(128) → BN → Dropout(0.3)
           → Dense(64)  → BN → Dropout(0.2)
           → Dense(32)  → BN → Dropout(0.2)
           → Dense(1, sigmoid)

Total Parameters: 145,793
```
- Framework: TensorFlow/Keras
- Optimizer: Adam (lr=0.001)
- Callbacks: Early Stopping + ReduceLROnPlateau
- Class weights applied during training

---

## Results & Performance

### Model Comparison Table

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|:--------:|:---------:|:------:|:--------:|:-------:|
| Logistic Regression | 82.43% | 13.42% | 73.72% | 22.70% | 86.04% |
| Random Forest | 92.91% | 28.97% | 70.65% | 41.09% | 91.10% |
| **XGBoost** | 88.47% | 20.49% | **79.70%** | 32.60% | **92.02%** |
| Neural Network | **93.15%** | **29.48%** | 68.84% | **41.28%** | 91.04% |

### Best Model per Metric

| Metric | Best Model | Score |
|--------|------------|-------|
| **Accuracy** | Neural Network | 93.15% |
| **Precision** | Neural Network | 29.48% |
| **Recall** | XGBoost | 79.70% |
| **F1-Score** | Neural Network | 41.28% |
| **ROC-AUC** | XGBoost | 92.02% |

### Selected Best Model: **XGBoost**

**Alasan pemilihan:**
1. **ROC-AUC tertinggi (92.02%)** - Kemampuan diskriminasi terbaik
2. **Recall tertinggi (79.70%)** - Mendeteksi ~80% dari semua fraud
3. **Memory & Speed efficient** - Tidak memerlukan GPU
4. **Interpretable** - Feature importance tersedia

### Top 5 Most Important Features (XGBoost)

| Rank | Feature | Importance |
|------|---------|------------|
| 1 | V258 | 10.98% |
| 2 | V70 | 8.29% |
| 3 | V91 | 5.87% |
| 4 | V201 | 5.40% |
| 5 | V294 | 3.66% |

---

## Key Insights

### Mengapa Recall Lebih Penting dari Precision?

Dalam **fraud detection**, konsekuensi missing fraud jauh lebih besar daripada false positive:

```
Cost of Missing Fraud >> Cost of False Positive
```

| Scenario | Consequence |
|----------|-------------|
| **False Positive** | Customer minor inconvenience, quick verification |
| **False Negative (Missed Fraud)** | Financial loss, reputation damage, legal issues |

### Trade-off Analysis

| Model Type | Pros | Cons |
|------------|------|------|
| **High Recall (XGBoost)** | 80% fraud tertangkap | Lebih banyak false alarm |
| **High Precision (Neural Network)** | Akurat saat prediksi fraud | Beberapa fraud terlewat |

### Recommendation per Use Case

| Use Case | Recommended Model | Threshold |
|----------|-------------------|-----------|
| **High Security (Banking)** | XGBoost | 0.3 (more detection) |
| **Balanced Approach** | XGBoost/Neural Network | 0.5 |
| **Low False Positive** | Neural Network | 0.7 (higher precision) |

---

## Conclusion

### Summary

Pipeline fraud detection ini berhasil dibangun dengan hasil sebagai berikut:

- **4 model** dibandingkan secara komprehensif (Logistic Regression, Random Forest, XGBoost, Neural Network)

- **ROC-AUC > 90%** untuk 3 model terbaik, menunjukkan kemampuan diskriminasi yang excellent

- **Recall ~80%** untuk XGBoost, berhasil mendeteksi mayoritas transaksi fraud

- **Memory-efficient approach** berhasil menangani dataset besar tanpa memory issues

- **Prediksi test set** tersimpan dalam `fraud_predictions.csv` (506,691 predictions)

### Final Selected Model

| Attribute | Value |
|-----------|-------|
| **Model** | XGBoost |
| **ROC-AUC** | 92.02% |
| **Recall** | 79.70% |
| **F1-Score** | 32.60% |
| **Predictions** | 506,691 transactions |

### Future Improvements

1. **Feature Engineering**: Menambah fitur dari domain knowledge
2. **Ensemble Methods**: Kombinasi XGBoost + Neural Network
3. **Threshold Tuning**: Berdasarkan cost-benefit analysis
4. **Regular Retraining**: Dengan data baru untuk adaptasi pattern fraud

---

## How to Run

### Prerequisites
```bash
pip install polars pandas numpy matplotlib seaborn scikit-learn xgboost tensorflow
```

### Run the Notebook
1. Buka `finalterm_transaction_code.ipynb` di Jupyter Notebook/VS Code
2. Pastikan dataset tersedia di folder `finalterm_folder/`:
   - `train_transaction.csv`
   - `test_transaction.csv`
3. Jalankan semua cell secara berurutan

### Output
- `fraud_predictions.csv`: File prediksi dengan kolom `TransactionID` dan `isFraud` (probability)

---

## File Structure

```
UAS Dataset 1/
├── finalterm_transaction_code.ipynb  # Main notebook
├── fraud_predictions.csv             # Output predictions
├── README.md                         # This file
└── finalterm_folder/
    ├── train_transaction.csv         # Training data
    └── test_transaction.csv          # Test data
```

---

## Technical Notes

### Memory Optimization Strategies Used

| Strategy | Benefit |
|----------|---------|
| **Polars** instead of Pandas | Faster data manipulation, less memory |
| **Class Weights** instead of SMOTE | No memory explosion from synthetic data |
| **n_jobs=1** | Single thread, predictable memory usage |
| **No GridSearchCV** | Direct training dengan good defaults |

### Libraries Used

| Category | Libraries |
|----------|-----------|
| **Data Manipulation** | Polars, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow/Keras |

---
