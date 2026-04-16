from core.preprocess import preprocess
from core.alignment import align
from detection.yolo_detector import YOLODetector
from comparison.ssim import compute_ssim
from comparison.defect_fusion import fuse_defects
from core.roi_defect_engine import detect_roi_defects
import json
from config import PCB_DATABASE
import cv2

yolo = YOLODetector("models/best.pt")

def run_pipeline(image, pcb_type):

    image = preprocess(image)
    golden = cv2.imread(PCB_DATABASE[pcb_type])

    aligned, H, match_score = align(image, golden)

    detections = yolo.detect(aligned)

    ssim_score = compute_ssim(aligned, golden)


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

    # LOAD ROI MAP
    with open("backend/data/roi_map.json", "r") as f:
    roi_map = json.load(f)

    # 🔥 NEW: ROI-LEVEL DEFECTS
    roi_defects = detect_roi_defects(detections, roi_map)

    # 6. FUSION ENGINE (IMPORTANT)
    defects = fuse_defects(
        detections=detections,
        ssim_score=ssim_score,
        match_score=match_score
    )

    return {
        "status": "FAIL" if (roi_defects or fusion["defects"]) else "PASS",
        "roi_defects": roi_defects,
        "fusion_defects": fusion["defects"],
        "detections": detections,
        "ssim": ssim_score,
        "alignment": match_score
    }
