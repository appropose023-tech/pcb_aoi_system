import json
import cv2
from detection.yolo_detector import YOLODetector

yolo = YOLODetector()

image_path = "backend/data/golden_pcb.jpg"
output_path = "backend/data/roi_map_auto.json"

img = cv2.imread(image_path)

detections = yolo.detect(img)

roi_map = {}
class_prefix = {"0": "C", "1": "R", "2": "Co", "3": "IC", "4": "D", "5": "r", "6": L", "7": "c"}
counts = {}

for d in detections:
    cls = d["cls"]
    box = d["box"]

    prefix = class_prefix.get(cls, "U")
    counts[prefix] = counts.get(prefix, 0) + 1

    name = f"{prefix}{counts[prefix]}"

    roi_map[name] = {
        "box": [int(x) for x in box],
        "class": cls
    }

with open(output_path, "w") as f:
    json.dump(roi_map, f, indent=4)

print("Auto ROI generated!")
