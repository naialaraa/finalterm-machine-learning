# Summary: Online Transaction Fraud Detection

## Project Overview

End-to-End Machine Learning & Deep Learning Pipeline for Fraud Detection  

---

## Dataset Information

| Keterangan | Training Data | Test Data |
|------------|---------------|-----------|
| Jumlah Transaksi | 590,540 | 506,691 |
| Jumlah Fitur | 393 | 393 |
| Target Variable | isFraud (0/1) | - |

### Class Distribution (Imbalanced)
- **Not Fraud (0):** ~96.5% 
- **Fraud (1):** ~3.5%
- **Imbalance Ratio:** 1:27

---

## Preprocessing Steps

1. **Missing Value Handling:**
   - Numerical features: Imputed with **median**
   - Categorical features: Imputed with **'Unknown'**

2. **Encoding:**
   - Label Encoding untuk 6 categorical columns

3. **Feature Scaling:**
   - StandardScaler untuk normalisasi semua fitur

4. **Train-Validation Split:**
   - Training: 80% (472,432 samples)
   - Validation: 20% (118,108 samples)
   - Stratified split untuk menjaga distribusi class

5. **Class Imbalance Handling:**
   - Menggunakan **Class Weights** (bukan SMOTE)
   - Fraud class diberi bobot ~14x lebih tinggi
   - Memory-efficient untuk dataset besar

---

## Models Trained

### 1. Logistic Regression (Traditional ML)
- Linear model untuk baseline
- Parameter: C=0.1, solver='lbfgs'
- class_weight='balanced'

### 2. Random Forest (Ensemble ML)
- 100 trees, max_depth=15
- class_weight='balanced'
- Handles non-linear relationships

### 3. XGBoost (Gradient Boosting)
- 150 boosting rounds, max_depth=6
- scale_pos_weight untuk class imbalance
- State-of-the-art untuk tabular data

### 4. Neural Network - MLP (Deep Learning)
- Framework: TensorFlow/Keras
- Architecture: 256 - 128 - 64 - 32 - 1
- BatchNormalization + Dropout (0.2-0.3)
- Optimizer: Adam (lr=0.001)
- Early Stopping + ReduceLROnPlateau
- Total Parameters: 145,793

---

## Model Performance Comparison

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|-------|----------|-----------|--------|----------|---------|
| Logistic Regression | 82.43% | 13.42% | 73.72% | 22.70% | 86.04% |
| Random Forest | 92.91% | 28.97% | 70.65% | 41.09% | 91.10% |
| **XGBoost** | 88.47% | 20.49% | **79.70%** | 32.60% | **92.02%** |
| Neural Network (TensorFlow) | **93.15%** | **29.48%** | 68.84% | **41.28%** | 91.04% |

### Best Model per Metric:
- **Accuracy:** Neural Network (93.15%)
- **Precision:** Neural Network (29.48%)
- **Recall:** XGBoost (79.70%) [BEST]
- **F1-Score:** Neural Network (41.28%)
- **ROC-AUC:** XGBoost (92.02%) [BEST]

---

## Best Model Selection

### Winner: XGBoost

**Alasan:**
1. **ROC-AUC tertinggi (92.02%)** - Kemampuan diskriminasi terbaik
2. **Recall tertinggi (79.70%)** - Mendeteksi ~80% dari semua fraud
3. **Memory & Speed efficient** - Tidak memerlukan GPU
4. **Interpretable** - Feature importance tersedia

### Top 5 Most Important Features:
1. V258 (10.98%)
2. V70 (8.29%)
3. V91 (5.87%)
4. V201 (5.40%)
5. V294 (3.66%)

---

## Key Insights

### Mengapa Recall Lebih Penting dari Precision?

Dalam **fraud detection**, lebih baik:
- Menangkap 80 fraud dari 100 (high recall) + investigasi 70 false positive
- Daripada hanya menangkap 30 fraud tapi precision tinggi

**Cost of Missing Fraud >> Cost of False Positive**

### Trade-off Analysis:
- **High Recall Model (XGBoost):** Lebih banyak fraud tertangkap, tapi lebih banyak false alarm
- **High Precision Model (Neural Network):** Lebih akurat saat prediksi fraud, tapi beberapa fraud terlewat

---

## Output

File prediksi disimpan di: `fraud_predictions.csv`

| Column | Description |
|--------|-------------|
| TransactionID | ID unik transaksi |
| isFraud | Probabilitas fraud (0-1) |

Total Predictions: **506,691 transactions**

---

## Recommendations

### Untuk Production:
1. Gunakan **XGBoost** sebagai primary model
2. Set threshold berdasarkan business requirement:
   - Threshold 0.3: Higher recall, more investigation
   - Threshold 0.5: Balanced approach
   - Threshold 0.7: Higher precision, miss some fraud

### Untuk Improvement:
1. Feature engineering dari domain knowledge
2. Ensemble: Combine XGBoost + Neural Network
3. Tune threshold berdasarkan cost-benefit analysis
4. Regular retraining dengan data baru

---

## Technical Notes

### Memory Optimization:
- **Polars** instead of Pandas untuk data manipulation
- **Class Weights** instead of SMOTE (no memory explosion)
- **n_jobs=1** untuk model training (single thread)
- **No GridSearchCV** (direct training dengan good defaults)

### Libraries Used:
- polars, pandas, numpy
- scikit-learn
- xgboost
- tensorflow/keras
- matplotlib, seaborn

---

## Conclusion

Pipeline fraud detection ini berhasil dibangun dengan:
- 4 model yang dibandingkan secara komprehensif
- ROC-AUC > 90% untuk 3 model terbaik
- Recall ~80% untuk XGBoost (menangkap mayoritas fraud)
- Memory-efficient approach untuk large dataset
- Prediksi test set tersimpan dalam CSV

**Model terpilih: XGBoost** dengan ROC-AUC 92.02% dan Recall 79.70%

---

*Summary generated from: finalterm_transaction_data.ipynb*

