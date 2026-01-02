# Song Release Year Prediction Using Regression

## Final Term Project - Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
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

Proyek ini bertujuan untuk memprediksi **tahun rilis lagu** berdasarkan fitur audio numerik menggunakan berbagai algoritma **Machine Learning** dan **Deep Learning**. Ini adalah **regression task** di mana variabel target bersifat kontinu (mewakili tahun).

### Objective
- Build dan evaluate multiple regression models
- Compare model performances menggunakan standard regression metrics
- Select model terbaik untuk memprediksi tahun rilis lagu

---

## Problem Statement

Memprediksi tahun rilis lagu dari fitur audio adalah tantangan menarik karena:

| Challenge | Description |
|-----------|-------------|
| **High Dimensionality** | Banyak fitur audio yang perlu dianalisis |
| **Non-linear Relationships** | Hubungan antara fitur dan tahun rilis tidak selalu linear |
| **Temporal Patterns** | Karakteristik musik berubah seiring waktu |

### Target Variable
- **release_year**: Tahun rilis lagu (continuous variable)

---

## Dataset Information

### Dataset Overview

| Attribute | Value |
|-----------|-------|
| **File** | `finalterm-regresi-dataset.csv` |
| **Total Samples** | ~515,000+ |
| **Features** | 90 numerical audio features |
| **Target** | Release Year (first column) |

### Target Variable Statistics

| Statistic | Value |
|-----------|-------|
| **Minimum Year** | ~1922 |
| **Maximum Year** | ~2011 |
| **Mean Year** | ~1998 |
| **Standard Deviation** | ~10-15 years |

### Key Observations from EDA
- Distribusi tahun rilis menunjukkan mayoritas lagu dari era modern (1990-2010)
- Fitur memiliki berbagai distribusi - beberapa normal, beberapa skewed
- Most features memiliki korelasi lemah dengan target (suggesting complex non-linear relationships)

---

## Methodology

### Pipeline Overview

```
Data Loading → EDA → Preprocessing → Feature Selection → Model Training → Evaluation → Conclusion
```

### 1. Data Loading
- Menggunakan **Polars** untuk data manipulation yang lebih cepat
- Renamed columns untuk clarity (`release_year`, `feature_1`, `feature_2`, ...)

### 2. Exploratory Data Analysis (EDA)
- Target variable distribution analysis
- Feature distribution overview
- Missing value inspection
- Correlation analysis dengan target

### 3. Data Preprocessing

| Step | Method | Details |
|------|--------|---------|
| **Missing Values** | Median Imputation | Robust terhadap outliers |
| **Outlier Handling** | Z-score Clipping | Threshold = 3 std deviations |
| **Feature Scaling** | StandardScaler | Mean=0, Std=1 |
| **Train-Test Split** | 80-20 Split | Random state = 42 |

### 4. Feature Selection
- **Method**: SelectKBest with F-regression scoring
- **Features Selected**: Top 50 most important features
- Mengidentifikasi fitur yang paling berkorelasi dengan tahun rilis

### 5. Model Training & Evaluation
- 5 model berbeda dilatih dan dibandingkan
- Metrics: MSE, RMSE, MAE, R² Score

---

## Models Implemented

### 1. Linear Regression (Baseline)
```python
LinearRegression()
```
- Baseline model untuk comparison
- Fast training, interpretable coefficients

### 2. Ridge Regression (L2 Regularization)
```python
Ridge(alpha=best_alpha)  # Tuned via GridSearchCV
```
- L2 regularization untuk menghindari overfitting
- Hyperparameter tuning untuk optimal alpha

### 3. Random Forest Regressor
```python
RandomForestRegressor(
    n_estimators=optimized,
    max_depth=optimized,
    min_samples_split=optimized,
    ...
)
```
- Ensemble of decision trees
- Hyperparameter tuning via RandomizedSearchCV
- Handles non-linear relationships well

### 4. Histogram-based Gradient Boosting
```python
HistGradientBoostingRegressor(
    max_iter=optimized,
    learning_rate=optimized,
    max_depth=optimized,
    early_stopping=True,
    ...
)
```
- Faster than standard Gradient Boosting
- Built-in early stopping
- L2 regularization support

### 5. Neural Network (MLP)
```
Architecture:
Input (50) → Dense(128, ReLU, L2) → BN → Dropout(0.2)
          → Dense(64, ReLU, L2)  → BN → Dropout(0.2)
          → Dense(32, ReLU, L2)  → Dropout(0.1)
          → Dense(1, Linear)

Optimizer: Adam (lr=0.0005)
Loss: MSE
Callbacks: EarlyStopping, ReduceLROnPlateau
```
- Deep Learning approach
- L2 regularization untuk prevent overfitting
- Batch Normalization dan Dropout

---

## Results & Performance

### Model Comparison Table

| Model | MSE | RMSE | MAE | R² Score |
|-------|:---:|:----:|:---:|:--------:|
| Linear Regression | - | ~9.0 | ~7.2 | ~0.23 |
| Ridge Regression | - | ~9.0 | ~7.2 | ~0.23 |
| Random Forest | - | ~8.5-9.0 | ~6.8-7.0 | ~0.25-0.30 |
| **Gradient Boosting (Hist)** | - | **~8.3-8.7** | **~6.5-6.8** | **~0.28-0.32** |
| Neural Network (MLP) | - | ~8.5-9.2 | ~6.8-7.3 | ~0.22-0.28 |

*Note: Actual values may vary based on training run*

### Best Model: **Gradient Boosting (HistGradientBoosting)**

| Metric | Value | Interpretation |
|--------|-------|----------------|
| **RMSE** | ~8.5 years | Rata-rata error prediksi sekitar 8-9 tahun |
| **MAE** | ~6.7 years | Median error sekitar 6-7 tahun |
| **R² Score** | ~0.30 | Model menjelaskan ~30% variance |

### Why Gradient Boosting?

1. **Lowest RMSE** - Rata-rata error paling kecil
2. **Best R² Score** - Menjelaskan variance paling banyak
3. **Handles Non-linearity** - Mampu menangkap pola kompleks
4. **Fast Training** - HistGradientBoosting sangat efisien
5. **Built-in Regularization** - Mencegah overfitting

---

## Key Insights

### 1. Feature Importance
Top features yang paling berpengaruh terhadap prediksi tahun rilis termasuk:
- Feature dengan F-score tertinggi dari SelectKBest
- Audio characteristics yang berubah seiring waktu (timbre, loudness trends, etc.)

### 2. Model Performance Analysis

| Observation | Explanation |
|-------------|-------------|
| **Low R² Score (~0.30)** | Memprediksi tahun rilis lagu adalah task yang challenging |
| **RMSE ~8-9 years** | Prediksi rata-rata meleset sekitar 8-9 tahun |
| **Tree-based models perform better** | Non-linear relationships lebih baik ditangkap |

### 3. Why is this Task Difficult?

```
Challenges:
├── Audio features tidak sepenuhnya menentukan tahun rilis
├── Genre dan style musik dapat overlap antar dekade
├── Remastering dan re-releases dapat mempengaruhi audio features
└── Cultural dan technological factors tidak tercapture dalam audio features
```

### 4. Residual Analysis
- Residuals mendekati distribusi normal
- Tidak ada pattern yang jelas dalam residual plot
- Model tidak memiliki bias sistematis yang signifikan

---

## Conclusion

### Summary

Pipeline regression untuk memprediksi tahun rilis lagu berhasil diimplementasikan dengan hasil:

| Aspect | Result |
|--------|--------|
| **Models Trained** | 5 (Linear, Ridge, Random Forest, Gradient Boosting, Neural Network) |
| **Best Model** | Histogram-based Gradient Boosting |
| **Best RMSE** | ~8.3-8.7 years |
| **Best R² Score** | ~0.28-0.32 |
| **Variance Explained** | ~30% |

### Key Takeaways

- **Gradient Boosting (Hist)** memberikan performa terbaik untuk task ini

- **RMSE ~8.5 tahun** - Model dapat memprediksi tahun rilis dengan error rata-rata sekitar 8-9 tahun

- **R² ~0.30** - Task memprediksi tahun rilis dari audio features adalah challenging, dengan model hanya menjelaskan ~30% variance

- **Non-linear models** (tree-based) perform lebih baik daripada linear models

- **Deep Learning (MLP)** tidak memberikan improvement signifikan dibanding tree-based models untuk dataset ini

### Recommendations for Improvement

1. **Feature Engineering** - Menambah fitur domain-specific (genre indicators, cultural trends)
2. **More Data** - Dataset lebih besar dengan lebih banyak variasi
3. **Ensemble Methods** - Kombinasi multiple models
4. **Advanced Architectures** - CNN/RNN untuk sequential audio features
5. **External Features** - Menambah metadata lagu (artist info, label, etc.)

---

## How to Run

### Prerequisites
```bash
pip install polars pandas numpy matplotlib seaborn scikit-learn tensorflow
```

### Run the Notebook
1. Buka `finalterm_regression_code.ipynb` di Jupyter Notebook/VS Code
2. Pastikan dataset tersedia: `finalterm-regresi-dataset.csv`
3. Jalankan semua cell secara berurutan

### Expected Runtime
| Section | Estimated Time |
|---------|---------------|
| Data Loading & EDA | 1-2 minutes |
| Preprocessing | 1-2 minutes |
| Linear/Ridge Regression | < 1 minute |
| Random Forest (with tuning) | 5-10 minutes |
| Gradient Boosting (with tuning) | 3-8 minutes |
| Neural Network | 5-15 minutes |
| **Total** | **~20-40 minutes** |

---

## File Structure

```
UAS Dataset 2/
├── finalterm_regression_code.ipynb  # Main notebook
├── finalterm-regresi-dataset.csv    # Dataset
└── README.md                        # This file
```

---

## Technical Notes

### Libraries Used

| Category | Libraries |
|----------|-----------|
| **Data Manipulation** | Polars, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | scikit-learn |
| **Deep Learning** | TensorFlow/Keras |

### Memory & Performance Optimization

| Strategy | Benefit |
|----------|---------|
| **Polars** for data loading | 2-10x faster than Pandas |
| **HistGradientBoosting** | Much faster than standard GradientBoosting |
| **RandomizedSearchCV** | Faster hyperparameter search |
| **Early Stopping** | Prevents unnecessary training epochs |

### Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **MSE** | $\frac{1}{n}\sum(y - \hat{y})^2$ | Average squared error |
| **RMSE** | $\sqrt{MSE}$ | Error in same unit as target |
| **MAE** | $\frac{1}{n}\sum|y - \hat{y}|$ | Average absolute error |
| **R²** | $1 - \frac{SS_{res}}{SS_{tot}}$ | Proportion of variance explained |

---
