import numpy as np

def extract_roi(image, box, margin=10):

    x1, y1, x2, y2 = map(int, box)

    h, w = image.shape[:2]

    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image[y1:y2, x1:x2]
