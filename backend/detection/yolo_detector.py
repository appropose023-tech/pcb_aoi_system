from ultralytics import YOLO

class YOLODetector:

    def __init__(self, model_path):
        self.model = YOLO(model_path)

    def detect(self, img):

        results = self.model(img, conf=0.4)[0]

        detections = []

        for box in results.boxes:
            detections.append({
                "cls": int(box.cls[0]),
                "conf": float(box.conf[0]),
                "box": box.xyxy[0].tolist()
            })

        return detections
