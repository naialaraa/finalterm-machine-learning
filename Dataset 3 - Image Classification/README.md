# Fish Image Classification Using Convolutional Neural Networks (CNN)

## Final Term Project - Machine Learning

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-orange.svg)](https://www.tensorflow.org/)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-red.svg)](https://keras.io/)

---

## Table of Contents
- [Project Overview](#-project-overview)
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

Proyek ini membangun pipeline **Image Classification** end-to-end untuk mengklasifikasikan **31 spesies ikan** menggunakan **Convolutional Neural Networks (CNN)**. Dua pendekatan utama dibandingkan:

1. **CNN from Scratch** - Model custom yang dibangun dari awal
2. **Transfer Learning** - Menggunakan pre-trained MobileNetV2

### Objectives
- Build dan evaluate CNN model dari scratch
- Implement Transfer Learning dengan MobileNetV2
- Compare model performances
- Visualize dan interpret CNN features

### Why CNN for Image Classification?

| Advantage | Description |
|-----------|-------------|
| **Automatic Feature Learning** | Belajar hierarchical features dari raw pixels |
| **Spatial Pattern Detection** | Convolutional layers mendeteksi edges, textures, shapes |
| **Translation Invariance** | Pooling layers memberikan robustness terhadap posisi |
| **Parameter Efficiency** | Lebih sedikit parameters dibanding fully connected networks |

---

## Dataset Information

### Dataset Overview

| Split | Images | Purpose |
|-------|--------|---------|
| **Training** | ~7,000+ | Model training dengan augmentation |
| **Validation** | ~1,500+ | Hyperparameter tuning |
| **Test** | ~1,500+ | Final evaluation |

### 31 Fish Species (Classes)

```
1. Bangus                    17. Janitor Fish
2. Big Head Carp             18. Knifefish
3. Black Spotted Barb        19. Long-Snouted Pipefish
4. Catfish                   20. Mosquito Fish
5. Climbing Perch            21. Mudfish
6. Fourfinger Threadfin      22. Mullet
7. Freshwater Eel            23. Pangasius
8. Glass Perchlet            24. Perch
9. Goby                      25. Scat Fish
10. Gold Fish                26. Silver Barb
11. Gourami                  27. Silver Carp
12. Grass Carp               28. Silver Perch
13. Green Spotted Puffer     29. Snakehead
14. Indian Carp              30. Tenpounder
15. Indo-Pacific Tarpon      31. Tilapia
16. Jaguar Gapote
```

### Image Properties
- **Resolution**: Variable (resized to 224x224 for training)
- **Format**: JPG/PNG
- **Channels**: RGB (3 channels)

---

## Methodology

### Pipeline Overview

```
Data Loading → EDA → Preprocessing → Augmentation → Model Training → Evaluation → Interpretation
```

### 1. Data Exploration
- Dataset structure verification
- Class distribution analysis
- Image dimension statistics
- Sample visualization

### 2. Data Preprocessing

| Step | Configuration |
|------|---------------|
| **Image Size** | 224 x 224 pixels |
| **Normalization** | Rescale to [0, 1] |
| **Batch Size** | 32 |
| **Input Shape** | (224, 224, 3) |

### 3. Data Augmentation (Training Only)

| Augmentation | Value |
|--------------|-------|
| Rotation | ±20 degrees |
| Width Shift | 20% |
| Height Shift | 20% |
| Shear | 0.2 |
| Zoom | 20% |
| Horizontal Flip | True |
| Brightness | [0.8, 1.2] |

### 4. Class Imbalance Handling
- **Method**: Computed Class Weights (`class_weight='balanced'`)
- Classes dengan lebih sedikit samples diberi bobot lebih tinggi

### 5. Training Callbacks

| Callback | Configuration |
|----------|---------------|
| **EarlyStopping** | patience=10, monitor='val_loss' |
| **ReduceLROnPlateau** | factor=0.2, patience=5 |
| **ModelCheckpoint** | Save best model by val_accuracy |

---

## Models Implemented

### 1. CNN from Scratch

```
Architecture:
┌─────────────────────────────────────────────────────────────┐
│ Block 1: Conv2D(32) → BN → Conv2D(32) → BN → MaxPool → Drop │
│ Block 2: Conv2D(64) → BN → Conv2D(64) → BN → MaxPool → Drop │
│ Block 3: Conv2D(128) → BN → Conv2D(128) → BN → MaxPool → Drop│
│ Block 4: Conv2D(256) → BN → Conv2D(256) → BN → MaxPool → Drop│
│ Flatten → Dense(512) → BN → Drop → Dense(256) → BN → Drop   │
│ Output: Dense(31, softmax)                                   │
└─────────────────────────────────────────────────────────────┘

Key Features:
- 4 Convolutional blocks (32→64→128→256 filters)
- BatchNormalization after each Conv layer
- MaxPooling2D for spatial reduction
- Dropout (0.25 in conv, 0.5 in dense) for regularization
- Total Parameters: ~3-5M
```

### 2. Transfer Learning (MobileNetV2)

```
Architecture:
┌─────────────────────────────────────────────────────────────┐
│ Base: MobileNetV2 (pre-trained on ImageNet, frozen)         │
│ ↓                                                           │
│ GlobalAveragePooling2D                                      │
│ ↓                                                           │
│ BatchNormalization                                          │
│ ↓                                                           │
│ Dense(256) → Dropout(0.5)                                   │
│ ↓                                                           │
│ Output: Dense(31, softmax)                                  │
└─────────────────────────────────────────────────────────────┘

Training Strategy:
- Phase 1: Frozen base, train only top layers (15 epochs)
- Phase 2: Fine-tune top 30 layers with lower LR (10 epochs)
```

### Why MobileNetV2?

| Advantage | Description |
|-----------|-------------|
| **Efficient** | Optimized untuk mobile/edge devices |
| **Pre-trained** | Learned features dari 1000+ ImageNet classes |
| **Fast Training** | Frozen base = fewer trainable parameters |
| **Strong Performance** | Competitive accuracy dengan fewer parameters |

---

## Results & Performance

### Model Comparison Table

| Metric | CNN from Scratch | Transfer Learning (MobileNetV2) |
|--------|:----------------:|:-------------------------------:|
| **Accuracy** | ~75-85% | ~90-95% |
| **Precision** | ~0.75-0.85 | ~0.90-0.95 |
| **Recall** | ~0.75-0.85 | ~0.90-0.95 |
| **F1-Score** | ~0.75-0.85 | ~0.90-0.95 |
| **Parameters** | ~3-5M | ~2.5M (mostly frozen) |
| **Training Time** | 20+ hours | ~1 hour |

*Note: Actual values may vary based on training run*

### Best Model: **Transfer Learning (MobileNetV2)**

| Aspect | Value |
|--------|-------|
| **Final Accuracy** | ~90-95% |
| **F1-Score** | ~0.90-0.95 |
| **Training Time** | ~30-60 minutes |
| **Model Size** | ~10-15 MB |

### Why Transfer Learning Wins?

1. **Pre-trained Features** - MobileNetV2 sudah belajar visual features dari ImageNet
2. **Faster Training** - Hanya train top layers (frozen base)
3. **Better Generalization** - Learned features lebih robust
4. **Less Overfitting** - Pre-trained weights sebagai regularization

---

## Key Insights

### 1. CNN Feature Learning

CNN secara otomatis belajar hierarchical features:

| Layer Level | Features Learned |
|-------------|------------------|
| **Early layers** | Edges, colors, simple textures |
| **Middle layers** | Shapes, patterns, fish body parts |
| **Deep layers** | Fish species-specific features |

### 2. Data Augmentation Impact

| With Augmentation | Without Augmentation |
|-------------------|----------------------|
| Better generalization | Overfitting risk |
| Robust to variations | Poor on unseen data |
| Effective use of limited data | Requires more data |

### 3. Class Imbalance Effects

- Beberapa spesies memiliki lebih sedikit training samples
- **Class weights** membantu model tidak bias ke majority class
- Per-class accuracy analysis menunjukkan challenging classes

### 4. Challenging Fish Species

Beberapa spesies yang lebih sulit diklasifikasi:
- Spesies dengan visual similarity tinggi
- Spesies dengan variasi warna/pattern
- Spesies dengan posisi/orientasi berbeda

### 5. Confidence Analysis

| Prediction Type | Confidence Pattern |
|-----------------|-------------------|
| **Correct** | High confidence (>0.8) |
| **Incorrect** | Lower confidence, often confused with similar species |

---

## Conclusion

### Summary

Pipeline image classification untuk 31 spesies ikan berhasil diimplementasikan dengan hasil:

| Aspect | Result |
|--------|--------|
| **Models Trained** | 2 (CNN from Scratch, Transfer Learning) |
| **Best Model** | Transfer Learning (MobileNetV2) |
| **Best Accuracy** | ~90-95% |
| **Number of Classes** | 31 fish species |
| **Training Time** | ~1 hour (Transfer Learning) |

### Key Takeaways

- **Transfer Learning** secara signifikan outperform CNN from scratch

- **MobileNetV2** memberikan balance yang baik antara accuracy dan efficiency

- **Data Augmentation** crucial untuk generalization

- **Class Weights** membantu handle imbalanced dataset

- **Feature Map Visualization** membantu understand apa yang CNN pelajari

### Model Selection Recommendation

| Use Case | Recommended Model |
|----------|-------------------|
| **Production/Deployment** | Transfer Learning (MobileNetV2) |
| **Learning/Education** | CNN from Scratch |
| **Mobile Apps** | MobileNetV2 (optimized for mobile) |
| **Real-time Classification** | MobileNetV2 (fast inference) |

### Future Improvements

1. **More Data** - Collect more images per class
2. **Advanced Augmentation** - CutMix, MixUp, AutoAugment
3. **Other Architectures** - EfficientNet, ResNet, Vision Transformers
4. **Ensemble Methods** - Combine multiple models
5. **Grad-CAM** - Visual explanations untuk predictions
6. **Model Optimization** - Quantization untuk mobile deployment

---

## How to Run

### Prerequisites
```bash
pip install tensorflow numpy pandas matplotlib seaborn pillow scikit-learn
```

### Run the Notebook
1. Buka `finalterm-image-code.ipynb` di Jupyter Notebook/VS Code
2. Pastikan dataset tersedia di folder `FishImgDataset/`:
   ```
   FishImgDataset/
   ├── train/
   │   ├── Bangus/
   │   ├── Big Head Carp/
   │   └── ... (31 folders)
   ├── val/
   │   └── ... (31 folders)
   └── test/
       └── ... (31 folders)
   ```
3. Jalankan semua cell secara berurutan

### Expected Runtime

| Section | Estimated Time |
|---------|---------------|
| Data Loading & EDA | 2-5 minutes |
| CNN from Scratch Training | 20+ hours (GPU recommended) |
| Transfer Learning Training | 30-60 minutes |
| Evaluation & Visualization | 5-10 minutes |
| **Total (Transfer Learning only)** | **~1-2 hours** |

### Hardware Recommendations

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| **RAM** | 8 GB | 16 GB |
| **GPU** | Optional | NVIDIA GPU with CUDA |
| **Storage** | 5 GB | 10 GB |

---

## File Structure

```
UAS Dataset 3/
├── finalterm-image-code.ipynb    # Main notebook
├── best_cnn_model.keras          # Saved CNN model
├── best_transfer_model.keras     # Saved Transfer Learning model
├── README.md                     # This file
└── FishImgDataset/
    ├── train/                    # Training images (31 folders)
    ├── val/                      # Validation images (31 folders)
    └── test/                     # Test images (31 folders)
```

---

## Technical Notes

### Libraries Used

| Category | Libraries |
|----------|-----------|
| **Deep Learning** | TensorFlow, Keras |
| **Data Manipulation** | NumPy, Pandas |
| **Visualization** | Matplotlib, Seaborn, PIL |
| **Evaluation** | scikit-learn |

### CNN Architecture Components

| Component | Purpose |
|-----------|---------|
| **Conv2D** | Extract spatial features dari images |
| **MaxPooling2D** | Reduce spatial dimensions, provide translation invariance |
| **BatchNormalization** | Stabilize training, enable higher learning rates |
| **Dropout** | Prevent overfitting dengan randomly dropping neurons |
| **GlobalAveragePooling2D** | Reduce feature maps to vector (Transfer Learning) |
| **Softmax** | Multi-class probability distribution |

### Evaluation Metrics Explained

| Metric | Formula | Interpretation |
|--------|---------|----------------|
| **Accuracy** | $\frac{TP + TN}{Total}$ | Overall correct predictions |
| **Precision** | $\frac{TP}{TP + FP}$ | How many predicted positives are correct |
| **Recall** | $\frac{TP}{TP + FN}$ | How many actual positives were found |
| **F1-Score** | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Harmonic mean of P and R |

---
