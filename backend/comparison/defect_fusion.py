def fuse_defects(detections, ssim_score, match_score):

    defects = []

    # RULE 1: alignment failure
    if match_score < 0.3:
        defects.append("ALIGNMENT_FAILED")

    # RULE 2: global mismatch
    if ssim_score < 0.75:
        defects.append("GLOBAL_MISMATCH")

    # RULE 3: low confidence components
    for d in detections:
        if d["conf"] < 0.5:
            defects.append("LOW_CONF_COMPONENT")

    # RULE 4: missing component heuristic
    if len(detections) < 5:
        defects.append("POSSIBLE_MISSING_PARTS")

    return list(set(defects))
