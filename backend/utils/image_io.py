import cv2
import numpy as np

def read_image(image_bytes):
    img = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(img, cv2.IMREAD_COLOR)
    return img


def resize_fixed(img, size=1024):
    return cv2.resize(img, (size, size))


def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
