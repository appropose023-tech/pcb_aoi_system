import cv2
import numpy as np

def match_template(img, template):

    res = cv2.matchTemplate(img, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(res)

    return float(max_val)


def multi_template_check(img, templates):
    scores = []

    for t in templates:
        scores.append(match_template(img, t))

    return max(scores) if scores else 0.0
