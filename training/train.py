from ultralytics import YOLO

def train():
    model = YOLO("yolov8n.pt")

    model.train(
        data="training/dataset.yaml",
        epochs=100,
        imgsz=640,
        batch=16,
        workers=4,
        device=0
    )

if __name__ == "__main__":
    train()
