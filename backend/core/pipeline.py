from core.preprocess import preprocess
from core.alignment import align
from detection.yolo_detector import YOLODetector
from comparison.ssim import compute_ssim
from comparison.defect_fusion import fuse_defects
from config import PCB_DATABASE
import cv2

yolo = YOLODetector("models/best.pt")

def run_pipeline(image, pcb_type):

    # 1. PREPROCESS (FIXED)
    image = preprocess(image)

    # 2. LOAD GOLDEN
    golden = cv2.imread(PCB_DATABASE[pcb_type])

    # 3. ALIGNMENT
    aligned, H, match_score = align(image, golden)

    # 4. YOLO DETECTION
    detections = yolo.detect(aligned)

    # 5. SSIM GLOBAL CHECK
    ssim_score = compute_ssim(aligned, golden)

    # 6. FUSION ENGINE (IMPORTANT)
    defects = fuse_defects(
        detections=detections,
        ssim_score=ssim_score,
        match_score=match_score
    )

    return {
        "status": "FAIL" if len(defects) > 0 else "PASS",
        "ssim": float(ssim_score),
        "alignment_score": float(match_score),
        "detections": detections,
        "defects": defects
    }
