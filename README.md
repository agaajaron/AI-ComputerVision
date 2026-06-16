# AI-ComputerVision

A collection of computer vision projects using CNNs, transfer learning, object detection, and metric learning, built with TensorFlow/Keras and PyTorch.

## Projects

| Folder | Task | Model | Dataset |
|---|---|---|---|
| [flowers-cnn-hyperparameter-tuning](flowers-cnn-hyperparameter-tuning/) | 5-class flower classification | CNN + Keras Tuner RandomSearch | Flowers dataset (5 species, 100×100) |
| [foodimage-classification](foodimage-classification/) | 3-class food image classification | CNN, VGG16 TL | Food images (Bread, Soup, Veg-Fruit, 150×150) |
| [mnistobjectdetection](mnistobjectdetection/) | Digit detection + bounding box | VGG16 dual-head (classification + regression) | Custom MNIST detection (9k/1k, 300×300) |
| [mri-braintumor-transfer-learning](mri-braintumor-transfer-learning/) | Binary MRI tumor classification | Custom CNN, VGG16 TL | Kaggle Brain Tumor MRI (1000 images, 224×224) |
| [siamese-network-training](siamese-network-training/) | Similarity learning on digit pairs | Siamese CNN with contrastive loss | MNIST (60k/10k, 28×28) |
| [vehicle-object-detection-using-yolov5](vehicle-object-detection-using-yolov5/) | Multi-class vehicle detection | YOLOv5 (Ultralytics) | Vehicles-OpenImages (5 classes) |
| [vehicle-objectdetection-yolov5](vehicle-objectdetection-yolov5/) | Multi-class vehicle detection | YOLOv5 (Ultralytics) | Vehicles-OpenImages (5 classes) |

Also included: `monkey_species_classification_final.py` — standalone script for monkey species classification.

## Project Structure

Each project follows a standard layout:

```
<project-name>/
├── data/              # Dataset files (not tracked in git)
├── notebooks/         # Original Jupyter notebooks
├── src/               # Python script version (# %% cell-marker format)
├── models/            # Saved model artifacts (not tracked in git)
└── requirements.txt
```

## Tech Stack

- Python, TensorFlow/Keras
- keras-tuner (hyperparameter search)
- YOLOv5 / Ultralytics
- OpenCV, NumPy, Pandas, Matplotlib, Seaborn, Scikit-learn
