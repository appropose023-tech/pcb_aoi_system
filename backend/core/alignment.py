import cv2
import numpy as np

def align(img, golden):

    gray1 = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    gray2 = cv2.cvtColor(golden, cv2.COLOR_BGR2GRAY)

    orb = cv2.ORB_create(6000)

    kp1, des1 = orb.detectAndCompute(gray1, None)
    kp2, des2 = orb.detectAndCompute(gray2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = bf.match(des1, des2)

    matches = sorted(matches, key=lambda x: x.distance)

    good = matches[:80]

    if len(good) < 15:
        return img, None, 0.0

    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    H, _ = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    aligned = cv2.warpPerspective(img, H, (golden.shape[1], golden.shape[0]))

    match_score = len(good) / 80.0

    return aligned, H, match_score
