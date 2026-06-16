# MNIST Object Detection — VGG16 Dual-Head

Object detection on MNIST handwritten digits: simultaneous digit classification and bounding-box regression using a dual-head VGG16 network.

## Dataset

- Custom MNIST-based object detection dataset (zip via Google Drive)
- 9,000 train / 1,000 test images, 300 × 300 px, RGB
- 10 classes (digits 0–9)
- Labels: per-image `.txt` files with columns `label, xmin, xmax, ymin, ymax`

## Model: VGG16 Dual-Head (`get_model`)

**Architecture**:
```
Input (300×300×3)
  └── VGG16 (frozen, ImageNet weights, include_top=False)
      └── Flatten
          ├── Classification head: Dropout(0.3) → Dense(10, softmax)    ← digit label
          └── Regression head:    Dense(128→64→32→16→4, sigmoid)        ← bounding box
```

**Training**:
- Optimizer: SGD (lr=0.01, momentum=0.94)
- Loss: `sparse_categorical_crossentropy` (labels) + `mse` (bounding boxes)
- Epochs: 10 | Batch: 64 | Validation split: 10%

**Results**:
- Bounding box MAE ≈ 0.023 | MSE ≈ 0.001
- Label accuracy ≈ 67% (limited by low-quality ground-truth annotations)

## Project Structure

```
mnistobjectdetection/
├── data/
├── models/
├── notebooks/
│   └── MNISTObjectDetection.ipynb
├── src/
│   └── mnistobjectdetection.py
└── requirements.txt
```

## Key Techniques

- **Multi-task learning**: single backbone, two output heads (classification + regression)
- **Transfer Learning**: VGG16 frozen base (ImageNet weights)
- Bounding box coordinates normalized to [0, 1]; sigmoid output head
- `plot_boxes` and `plot_predictions` functions to visualize ground-truth and predicted boxes
