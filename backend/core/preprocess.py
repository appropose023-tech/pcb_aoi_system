import cv2
import numpy as np

def preprocess(img):

    # FIXED SIZE (VERY IMPORTANT)
    img = cv2.resize(img, (1024, 1024))

    # FIXED COLOR SPACE
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # LIGHT NORMALIZATION
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)

    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    l = clahe.apply(l)

    lab = cv2.merge((l, a, b))
    img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)

    return img
