from ultralytics import YOLO
from config import MODEL_PATH, YOLO_CONF_THRESHOLD

class YOLODetector:

    def __init__(self):
        self.model = YOLO(MODEL_PATH)

    def detect(self, img):

        results = self.model(img, conf=YOLO_CONF_THRESHOLD, verbose=False)[0]

        detections = []

        if results.boxes is None:
            return detections

        for box in results.boxes:

            conf = float(box.conf[0])

            if conf < YOLO_CONF_THRESHOLD:
                continue

            detections.append({
                "cls": int(box.cls[0]),
                "conf": conf,
                "box": box.xyxy[0].tolist()
            })

        # SORT for deterministic behavior
        detections.sort(key=lambda x: (x["cls"], x["conf"]))

        return detections
