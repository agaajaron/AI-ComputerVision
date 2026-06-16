# Vehicle Object Detection using YOLOv5

Multi-class vehicle object detector trained with YOLOv5 on the Vehicles-OpenImages dataset.

## Dataset

- **Source**: Vehicles-OpenImages dataset (downloaded via Roboflow/FiftyOne)
- 5 classes: **Ambulance**, **Bus**, **Car**, **Motorcycle**, **Truck**
- Label format: YOLO format per `.txt` file — `class center_x center_y width height` (normalized to [0, 1])
- Note: dataset contains some duplicate images (identified and removed in preprocessing)

## Model: YOLOv5 (Ultralytics)

- **Repository**: [Ultralytics YOLOv5](https://github.com/ultralytics/yolov5)
- Training via `yolov5/train.py` with a `data.yaml` config file
- Multiple model sizes available: YOLOv5n (nano), YOLOv5s (small), YOLOv5m (medium), YOLOv5l (large), YOLOv5x (extra-large)
- Larger models have more parameters and higher GPU memory requirements

**`data.yaml` config**:
```yaml
nc: 5
names: ['Ambulance', 'Bus', 'Car', 'Motorcycle', 'Truck']
```

## Project Structure

```
vehicle-object-detection-using-yolov5/
├── data/
├── models/
├── notebooks/
│   └── Vehicle_Object_Detection_using_YOLOv5.ipynb
├── src/
│   └── vehicle_object_detection_using_yolov5.py
└── requirements.txt
```

## Key Techniques

- **YOLOv5**: single-pass object detection — predicts bounding boxes and class probabilities in one forward pass
- YOLO-format bounding box conversion (`yolo2reqbox`) for visualization
- `plot_box` and `plot` helper functions to draw predicted boxes on sample images
- TensorBoard integration for training monitoring
- Results stored in versioned `results_N/` directories
