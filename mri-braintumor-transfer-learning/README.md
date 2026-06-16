# MRI Brain Tumor Transfer Learning

Binary MRI image classifier distinguishing Pituitary Tumor from No Tumor, comparing a custom CNN against VGG16 transfer learning.

## Dataset

- **Source**: [Kaggle Brain Tumor Classification (MRI)](https://www.kaggle.com/datasets/sartajbhuvaji/brain-tumor-classification-mri)
- 1,000 images used (830 train / 170 test)
  - Train: 395 No Tumor + 435 Pituitary Tumor
- Images: 224 × 224 px, RGB (resized via `flow_from_directory`)
- Binary labels: 0 = no_tumor, 1 = pituitary_tumor

## Models

| Model | Architecture | Train Acc | Val Acc |
|---|---|---|---|
| Custom CNN | Conv2D(64→32→32→16) + BatchNorm + Dropout(0.25) → Dense(64/32/1, sigmoid) | ~95% | ~72% (overfitting) |
| **VGG16 TL** | Frozen VGG16 (ImageNet, 224×224) → Dense(32→32→1, sigmoid) | **~98%** | **~91%** |

**Best model**: VGG16 transfer learning — converges faster (5 epochs vs 10) with significantly better generalization.

## Data Augmentation

`ImageDataGenerator` with: horizontal flip, height/width shift (0.1), rotation (20°), shear (0.1), zoom (0.1), rescale 1/255.

## Project Structure

```
mri-braintumor-transfer-learning/
├── data/
├── models/
├── notebooks/
│   └── MRI_braintumor_Transfer+Learning.ipynb
├── src/
│   └── mri_braintumor_transfer_learning.py
└── requirements.txt
```

## Key Techniques

- **Transfer Learning**: VGG16 (frozen ImageNet weights), custom binary classification head
- Data augmentation via `ImageDataGenerator` + `flow_from_directory` (auto-labels from folder names)
- Batch Normalization + Dropout to reduce overfitting in baseline CNN
- `plot_history` helper function to compare training vs validation accuracy curves
