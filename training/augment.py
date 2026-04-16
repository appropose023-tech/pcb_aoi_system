import cv2
import albumentations as A

augment = A.Compose([
    A.RandomBrightnessContrast(p=0.5),
    A.MotionBlur(p=0.3),
    A.GaussNoise(p=0.3),
    A.Rotate(limit=10, p=0.5),
    A.Perspective(scale=(0.05, 0.1), p=0.5),
    A.CLAHE(p=0.3)
])

def generate(img):
    return augment(image=img)["image"]
