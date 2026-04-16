from skimage.metrics import structural_similarity as ssim
import cv2

def compute_ssim(img1, img2):

    g1 = cv2.cvtColor(img1, cv2.COLOR_RGB2GRAY)
    g2 = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)

    score, _ = ssim(g1, g2, full=True)

    return score
