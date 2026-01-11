# Machine Learning Final Term Projects (UAS)

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-Latest-yellow.svg)](https://scikit-learn.org/)

> **Tugas Akhir Semester (UAS) - Machine Learning**  
> **Semester 7 - 2025/2026**

---

## Table of Contents

- [Overview](#overview)
- [Projects](#projects)
  - [Dataset 1: Transaction Fraud Detection](#dataset-1-transaction-fraud-detection)
  - [Dataset 2: Song Release Year Prediction](#dataset-2-song-release-year-prediction)
  - [Dataset 3: Fish Image Classification](#dataset-3-fish-image-classification)
- [Tech Stack](#tech-stack)
- [Key Achievements](#key-achievements)
- [Installation & Setup](#installation--setup)
- [Repository Structure](#repository-structure)
- [Author](#author)

---

## Overview

Repository ini berisi **3 proyek Machine Learning** yang dikerjakan sebagai tugas akhir semester (UAS). Setiap proyek mendemonstrasikan implementasi end-to-end machine learning pipeline, mulai dari exploratory data analysis hingga model deployment dan evaluation.

### Learning Objectives
- **Classification**: Binary classification untuk fraud detection  
- **Regression**: Prediksi nilai kontinu (tahun rilis lagu)  
- **Deep Learning**: CNN dan Transfer Learning untuk image classification  
- **Model Comparison**: Perbandingan multiple algorithms untuk setiap task  
- **Best Practices**: Data preprocessing, handling imbalanced data, model evaluation

---

## Projects

### Dataset 1: Transaction Fraud Detection

**Task**: Binary Classification  
**Domain**: Financial Technology (FinTech)

#### Problem Statement
Mendeteksi transaksi fraudulent dalam dataset online transaction yang **highly imbalanced** (fraud rate ~3.5%).

#### Key Challenges
- Extreme class imbalance (1:27 ratio)
- 590K+ training samples dengan 393 features
- Real-time prediction requirement

#### Models Implemented
1. **Logistic Regression** - Baseline model
2. **Random Forest** - Ensemble learning
3. **XGBoost** - Gradient boosting
4. **Deep Neural Network** - Multi-layer perceptron

#### Best Result
- **Model**: XGBoost
- **ROC-AUC Score**: ~0.95+
- **Recall (Fraud)**: ~85%+
- **Precision (Fraud)**: ~80%+

#### Techniques Used
- SMOTE for handling class imbalance
- Feature engineering & selection
- Stratified K-Fold cross-validation
- Threshold optimization
- Class weight balancing

**[View Full Documentation →](Dataset%201%20-%20Transactions)**

---

### Dataset 2: Song Release Year Prediction

**Task**: Regression  
**Domain**: Music Information Retrieval

#### Problem Statement
Memprediksi tahun rilis lagu berdasarkan 90 fitur audio numerik dari dataset dengan 515K+ samples.

#### Key Challenges
- High-dimensional feature space (90 features)
- Non-linear relationships antara audio features dan release year
- Temporal patterns dalam karakteristik musik
- Range tahun: 1922-2011

#### Models Implemented
1. **Linear Regression** - Baseline model
2. **Ridge Regression** - L2 regularization
3. **Lasso Regression** - L1 regularization + feature selection
4. **Random Forest Regressor** - Ensemble method
5. **XGBoost Regressor** - Gradient boosting
6. **Deep Neural Network** - Multi-layer regression

#### Best Result
- **Model**: XGBoost Regressor / Deep Neural Network
- **R² Score**: ~0.75+
- **MAE**: ~8-10 years
- **RMSE**: ~10-12 years

#### Techniques Used
- Feature scaling (StandardScaler)
- Median imputation for missing values
- Feature correlation analysis
- Hyperparameter tuning
- Ensemble model comparison

**[View Full Documentation →](Dataset%202%20-%20Regression)**

---

### Dataset 3: Fish Image Classification

**Task**: Multi-class Image Classification  
**Domain**: Computer Vision

#### Problem Statement
Mengklasifikasikan gambar ikan ke dalam **31 spesies** menggunakan Convolutional Neural Networks (CNN).

#### Key Challenges
- 31-class classification problem
- Limited training data (~7,000 images)
- Variable image dimensions
- Class imbalance
- Complex visual similarities antar spesies

#### Models Implemented
1. **CNN from Scratch** - Custom architecture dengan 4 conv blocks
2. **Transfer Learning** - Pre-trained MobileNetV2 + fine-tuning

#### Best Result
- **Model**: Transfer Learning (MobileNetV2)
- **Test Accuracy**: ~85-90%
- **Training Time**: ~1 hour (vs 20+ hours for CNN from scratch)
- **Parameters**: 2.3M (vs 8M+ for custom CNN)

#### Techniques Used
- Data augmentation (rotation, flip, zoom, brightness)
- Transfer learning with frozen base layers
- Fine-tuning top layers
- Class weight balancing
- Feature map visualization
- Callbacks (EarlyStopping, ReduceLROnPlateau)
- Batch normalization & dropout

#### Dataset Split
- **Training**: 7,000+ images
- **Validation**: 1,500+ images
- **Test**: 1,500+ images

**[View Full Documentation →](Dataset%203%20-%20Image%20Classification)**

---

## Tech Stack

### Core Libraries

| Category | Libraries |
|----------|-----------|
| **Data Manipulation** | Polars, Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn, Plotly |
| **Machine Learning** | scikit-learn, XGBoost |
| **Deep Learning** | TensorFlow, Keras |
| **Computer Vision** | PIL, OpenCV |
| **Model Evaluation** | scikit-learn metrics |

### Deep Learning Frameworks
```python
tensorflow==2.x
keras (integrated with TensorFlow)
```

### Machine Learning
```python
scikit-learn>=1.0
xgboost>=1.5
```

---

## Key Achievements

### Project Highlights

| Project | Achievement | Metric |
|---------|-------------|--------|
| **Fraud Detection** | High recall pada fraud class | ROC-AUC: 0.95+ |
| **Song Prediction** | Accurate year prediction | R²: 0.75+, MAE: ~8 years |
| **Fish Classification** | Multi-class accuracy | Accuracy: 85-90% |

### Technical Skills Demonstrated

**Data Processing**
- Handling large datasets (500K+ rows)
- Missing value imputation
- Feature engineering
- Data augmentation

**Model Development**
- Traditional ML algorithms
- Deep Neural Networks
- Convolutional Neural Networks
- Transfer Learning

**Model Evaluation**
- Classification metrics (Precision, Recall, F1, ROC-AUC)
- Regression metrics (R², MAE, RMSE)
- Confusion matrices
- Cross-validation

**Advanced Techniques**
- Class imbalance handling (SMOTE, class weights)
- Hyperparameter tuning
- Ensemble methods
- Model interpretation
- Feature importance analysis

---

## Installation & Setup

### Prerequisites
```bash
Python 3.8+
pip or conda
```

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/machine-learning-uas.git
cd machine-learning-uas
```

2. **Create virtual environment**
```bash
# Using venv
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Or using conda
conda create -n ml-uas python=3.8
conda activate ml-uas
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

### Required Packages
```txt
numpy>=1.21.0
pandas>=1.3.0
polars>=0.14.0
matplotlib>=3.4.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
tensorflow>=2.8.0
pillow>=8.3.0
imbalanced-learn>=0.8.0
```

### Running the Notebooks

Each project contains Jupyter notebooks that can be run sequentially:

**Dataset 1 - Fraud Detection:**
```bash
jupyter notebook "UAS Dataset 1/finalterm_transaction_code.ipynb"
```

**Dataset 2 - Song Prediction:**
```bash
jupyter notebook "UAS Dataset 2/finalterm_regression_code.ipynb"
```

**Dataset 3 - Fish Classification:**
```bash
jupyter notebook "UAS Dataset 3/finalterm-image-code.ipynb"
```

---

## Results Summary

### Comparative Performance

| Project | Best Model | Key Metric | Score |
|---------|-----------|------------|-------|
| **Fraud Detection** | XGBoost | ROC-AUC | 0.95+ |
| **Song Prediction** | XGBoost/DNN | R² Score | 0.75+ |
| **Fish Classification** | Transfer Learning | Accuracy | 85-90% |

### Model Comparison Insights

#### Classification (Fraud Detection)
- XGBoost outperforms traditional ML and DNN
- SMOTE significantly improves recall
- Threshold optimization crucial for business metrics

#### Regression (Song Prediction)
- XGBoost and DNN perform similarly
- Ridge/Lasso underperform on non-linear relationships
- Feature engineering opportunity exists

#### Image Classification (Fish)
- Transfer Learning >>> CNN from Scratch
- MobileNetV2 provides best accuracy-speed tradeoff
- Data augmentation essential for generalization

---

## Learning Outcomes

### Machine Learning Concepts Mastered

1. **Supervised Learning**
   - Classification (Binary & Multi-class)
   - Regression
   
2. **Deep Learning**
   - Neural Networks
   - Convolutional Neural Networks
   - Transfer Learning
   
3. **Model Evaluation**
   - Cross-validation
   - Metrics selection based on problem type
   - Confusion matrix analysis
   
4. **Data Engineering**
   - Feature scaling & normalization
   - Handling missing data
   - Class imbalance techniques
   
5. **Model Optimization**
   - Hyperparameter tuning
   - Regularization
   - Early stopping

---

## Key Insights

### Best Practices Applied

- **Always start with EDA** - Understanding data is crucial  
- **Handle class imbalance** - Use SMOTE, class weights, or sampling  
- **Try multiple models** - Don't settle on the first model  
- **Use proper validation** - Avoid data leakage with stratified splits  
- **Optimize for business metrics** - Not just accuracy  
- **Document everything** - Code, decisions, and results  
- **Visualize results** - Makes interpretation easier  

---

## Author

**Naia Lara Shafir Anwar**  
TK-46-GAB
Machine Learning

</div>


