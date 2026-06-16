# Siamese Network Training

Similarity learning on MNIST digit pairs using a Siamese network — trains two identical CNN branches to produce embeddings that are close for same-class pairs and far apart for different-class pairs.

## Dataset

- **Source**: `tf.keras.datasets.mnist` — 60,000 train / 10,000 test images, 28 × 28 px, grayscale
- Input: **pairs** of images labeled 1 (same class) or 0 (different class)
- Pair construction: `mark_paired_datasets()` generates positive and negative pairs from the raw MNIST split

## Model: Siamese CNN

**Shared CNN backbone** (`get_cnn_block`):
```
Conv2D → BatchNormalization → ReLU   (repeated per depth)
→ GlobalAveragePooling2D
→ Dense (embedding vector)
→ Reshape
```

Two identical branches share weights; their outputs are compared to produce a similarity score.

**Training — two phases**:
1. **Binary cross-entropy loss**: trains with `BinaryCrossentropy`, Adam (lr=0.001), 5 epochs, batch 64
2. **Contrastive loss**: custom `contrastive_loss(y_true, y_pred)` — `y * d² + (1−y) * max(m−d, 0)²` — Adam, 5 epochs, batch 64

**Results** (contrastive loss run):
- Loss ≈ 0.09 (decreasing, model learning)
- Accuracy ≈ 10% (low — model not converging well; identified as area for improvement)

## Project Structure

```
siamese-network-training/
├── data/
├── models/
├── notebooks/
│   └── Siamese_Network_Training.ipynb
├── src/
│   └── siamese_network_training.py
└── requirements.txt
```

## Key Techniques

- **Siamese network**: shared-weight twin CNN branches for similarity learning
- **Contrastive loss**: pulls same-class embeddings together, pushes different-class embeddings apart
- Custom pair generation (`mark_paired_datasets`) from standard classification datasets
- `itertools` used for pair enumeration
