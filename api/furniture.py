from ultralytics import YOLO
import os
import random
from PIL import Image

furniture_classes_list = [
    "bench", "chair", "couch", "potted plant", 
    "bed", "dining table", "clock", "vase"
]

complementary_items = {
    "bench": [("vase", 5), ("potted plant", 4), ("clock", 3), ("cabinet", 2), ("shelf", 1), ("sideboard", 1)],
    "chair": [("dining table", 5), ("potted plant", 4), ("vase", 3), ("desk", 2), ("shelf", 1), ("wing_chair", 1)],
    "couch": [("vase", 5), ("potted plant", 4), ("bench", 3), ("sideboard", 2), ("tv_bench", 1), ("chaise", 1)],
    "potted plant": [("bench", 5), ("couch", 4), ("dining table", 3), ("shelf", 2), ("cabinet", 1), ("desk", 1)],
    "bed": [("vase", 5), ("clock", 4), ("shelf", 3), ("cabinet", 2), ("sideboard", 1), ("sleeper", 1)],
    "dining table": [("chair", 5), ("vase", 4), ("potted plant", 3), ("sideboard", 2), ("cabinet", 1), ("bench", 1)],
    "clock": [("bench", 5), ("couch", 4), ("vase", 3), ("shelf", 2), ("desk", 1), ("tv_bench", 1)],
    "vase": [("couch", 5), ("dining table", 4), ("bench", 3), ("shelf", 2), ("desk", 1), ("sideboard", 1)]
}

# Load the YOLOv9 model
model = YOLO('yolov9s.pt')

def detect_furniture(image):
    # Run predictions
    results = model.predict(image)

    # Filter detected classes
    detected_classes = [model.names[int(cls)] for cls in results[0].boxes.cls]
    filtered_classes = [cls for cls in detected_classes if cls in furniture_classes_list]

    return filtered_classes


def get_top_complementary_items(detected_items):
    item_counts = {}
    for item in detected_items:
        if item in complementary_items:
            for comp_item, count in complementary_items[item]:
                if comp_item in item_counts:
                    item_counts[comp_item] += count
                else:
                    item_counts[comp_item] = count
    
    # Remove detected items from suggestions
    for item in detected_items:
        if item in item_counts:
            del item_counts[item]

    # Sort items by count and return the top 5
    sorted_items = sorted(item_counts.items(), key=lambda x: x[1], reverse=True)
    return [item[0] for item in sorted_items[:5]]