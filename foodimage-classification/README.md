# Food Image Classification

3-class food image classifier for a stock photography company, comparing a custom CNN against a deeper CNN and VGG16 transfer learning.

## Dataset

- **Source**: Provided via Google Drive zip (`Food_Data.zip`)
- Images: 150 × 150 px, RGB
- 3 classes: **Bread**, **Soup**, **Vegetable-Fruit**
- Pre-split into Training and Testing directories

## Models

| Model | Architecture | Test Accuracy |
|---|---|---|
| Model 1 (baseline CNN) | Conv2D(64→32→32) → Dense(100) → Dense(3, softmax) — SGD | ~73% |
| Model 2 (deeper CNN) | Conv2D(256→128→64→32) + Dropout(0.3/0.5) → Dense(64/32) → Dense(3) — Adam | No overfitting |
| **Model 3 (VGG16 TL)** | Frozen VGG16 (ImageNet, 150×150) → Dense(32×2, relu) → Dense(3, softmax) + Augmentation | **~87%** |

**Best model**: VGG16 transfer learning with data augmentation (~87% test accuracy).

## Data Augmentation (Model 3)

`ImageDataGenerator` with: horizontal flip, height/width shift (0.1), rotation (20°), shear (0.1), zoom (0.1), rescale 1/255.

## Project Structure

```
foodimage-classification/
├── data/
├── models/
├── notebooks/
│   └── FoodImage_classification.ipynb
├── src/
│   └── foodimage_classification.py
└── requirements.txt
```

## Key Techniques

- Custom CNN architectures with EarlyStopping and ModelCheckpoint callbacks
- **Transfer Learning**: VGG16 (frozen, ImageNet weights), custom Dense head
- `LabelBinarizer` for one-hot encoding
- Confusion matrix and classification report per class
