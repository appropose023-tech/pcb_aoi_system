from config import SSIM_FAIL_THRESHOLD

def fuse_defects(detections, ssim_score, alignment_score):

    defects = []
    evidence_score = {
        "alignment": 0.0,
        "global": 0.0,
        "confidence": 0.0,
        "density": 0.0,
        "anomaly": 0.0
    }

    # -------------------------
    # 1. ALIGNMENT (graded)
    # -------------------------
    if alignment_score < 0.2:
        evidence_score["alignment"] = 1.0
    elif alignment_score < 0.3:
        evidence_score["alignment"] = 0.7
    elif alignment_score < 0.4:
        evidence_score["alignment"] = 0.4

    # -------------------------
    # 2. GLOBAL SIMILARITY (graded)
    # -------------------------
    if ssim_score < 0.6:
        evidence_score["global"] = 1.0
    elif ssim_score < SSIM_FAIL_THRESHOLD:
        evidence_score["global"] = 0.6
    elif ssim_score < 0.85:
        evidence_score["global"] = 0.3

    # -------------------------
    # 3. LOW CONFIDENCE ANALYSIS
    # -------------------------
    low_conf_count = sum(1 for d in detections if d["conf"] < 0.5)

    if low_conf_count > 5:
        evidence_score["confidence"] = 1.0
    elif low_conf_count > 2:
        evidence_score["confidence"] = 0.6
    elif low_conf_count > 0:
        evidence_score["confidence"] = 0.3

    # -------------------------
    # 4. DETECTION DENSITY
    # -------------------------
    if len(detections) < 3:
        evidence_score["density"] = 1.0
    elif len(detections) < 5:
        evidence_score["density"] = 0.5

    # -------------------------
    # 5. CLASS ANOMALY (still valid)
    # -------------------------
    cls_count = {}
    for d in detections:
        cls_count[d["cls"]] = cls_count.get(d["cls"], 0) + 1

    anomaly_flag = any(v > 10 for v in cls_count.values())
    if anomaly_flag:
        evidence_score["anomaly"] = 1.0

    # -------------------------
    # DECISION LAYER (STABLE LOGIC)
    # -------------------------

    # Hard fail conditions
    if evidence_score["alignment"] >= 0.7:
        defects.append("ALIGNMENT_FAIL")

    if evidence_score["global"] >= 0.7:
        defects.append("GLOBAL_MISMATCH")

    if evidence_score["anomaly"] >= 1.0:
        defects.append("DETECTION_ANOMALY")

    # Soft aggregation (reduces noise)
    soft_score = (
        evidence_score["confidence"] +
        evidence_score["density"]
    ) / 2

    if soft_score >= 0.75:
        defects.append("COMPONENT_ISSUE")
    elif soft_score >= 0.45:
        defects.append("POTENTIAL_COMPONENT_ISSUE")

    return {
        "defects": list(set(defects)),
        "evidence": evidence_score,
        "status": "FAIL" if len(defects) > 0 else "PASS"
    }
