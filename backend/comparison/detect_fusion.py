from config import SSIM_FAIL_THRESHOLD

def fuse_defects(detections, ssim_score, alignment_score):

    defects = []

    # 1. Alignment failure
    if alignment_score < 0.3:
        defects.append("ALIGNMENT_FAIL")

    # 2. Global mismatch
    if ssim_score < SSIM_FAIL_THRESHOLD:
        defects.append("GLOBAL_MISMATCH")

    # 3. Missing component heuristic
    if len(detections) < 5:
        defects.append("POSSIBLE_MISSING_COMPONENT")

    # 4. Low confidence logic
    low_conf = [d for d in detections if d["conf"] < 0.5]
    if len(low_conf) > 0:
        defects.append("LOW_CONF_DETECTION")

    # 5. Duplicate / inconsistent detection guard
    cls_count = {}
    for d in detections:
        cls_count[d["cls"]] = cls_count.get(d["cls"], 0) + 1

    for k, v in cls_count.items():
        if v > 10:
            defects.append("OVER_DETECTION_ANOMALY")

    return list(set(defects))
