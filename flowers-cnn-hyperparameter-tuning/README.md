# Flowers CNN Hyperparameter Tuning

5-class flower image classifier using a CNN with automated hyperparameter search via Keras Tuner RandomSearch.

## Dataset

- **Source**: Flowers dataset (`dataset/flowers/train` and `dataset/flowers/test`)
- Images: 100 × 100 px, RGB
- 5 classes: daisy, dandelion, rose, sunflower, tulip
- Loaded via `ImageDataGenerator.flow_from_directory` (batch size 64)

## Model

**Architecture** (defined in `build_model(hp)`):
```
Conv2D(32, 3×3, relu) → MaxPool(2×2)
Conv2D(64, 3×3, relu) → MaxPool(2×2)
Conv2D(128, 3×3, relu) → MaxPool(2×2)
Flatten
Dense(hp.Choice([128, 256, 512]), relu)   ← tuned hyperparameter
Dense(5, softmax)
```

**Training**:
- Optimizer: Adam | Loss: categorical_crossentropy
- Epochs: 8 | Batch: 64

## Hyperparameter Tuning

- **Tuner**: `keras_tuner.RandomSearch`
- **Search space**: Dense layer size — `[128, 256, 512]`
- **Objective**: `val_accuracy`
- Max trials: 4

Best model saved to `models/hyper_tuned_model.h5`.

## Project Structure

```
flowers-cnn-hyperparameter-tuning/
├── data/
├── models/
│   └── hyper_tuned_model.h5   # Best model from tuner
├── notebooks/
│   └── flowers_cnn_hyperparameter_tuning.ipynb
├── src/
│   └── flowers_cnn_hyperparameter_tuning.py
└── requirements.txt
```

## Key Techniques

- **Keras Tuner RandomSearch** for automated Dense layer size selection
- `ImageDataGenerator` for data loading and rescaling
- Single-image prediction example included (sunflower sample)
