import json

def get_center(box):
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2


def is_inside(center, roi_box):
    x, y = center
    x1, y1, x2, y2 = roi_box
    return x1 <= x <= x2 and y1 <= y <= y2


def detect_roi_defects(detections, roi_map):

    defects = []
    matched_rois = set()

    # STEP 1: assign detections to ROI
    roi_hits = {}

    for roi_name, roi_data in roi_map.items():
        roi_hits[roi_name] = None

        for det in detections:
            center = get_center(det["box"])

            if is_inside(center, roi_data["box"]):
                roi_hits[roi_name] = det
                matched_rois.add(roi_name)
                break

    # STEP 2: missing component detection
    for roi_name, det in roi_hits.items():
        if det is None:
            defects.append(f"{roi_name}_MISSING")

    # STEP 3: wrong class detection
    for roi_name, det in roi_hits.items():
        if det is not None:
            expected_class = roi_map[roi_name]["class"]

            if det["cls"] != expected_class:
                defects.append(f"{roi_name}_WRONG_COMPONENT")

    # STEP 4: shifted component detection (approx)
    for roi_name, det in roi_hits.items():
        if det is not None:
            cx, cy = get_center(det["box"])
            rx1, ry1, rx2, ry2 = roi_map[roi_name]["box"]

            roi_center_x = (rx1 + rx2) / 2
            roi_center_y = (ry1 + ry2) / 2

            dist = ((cx - roi_center_x)**2 + (cy - roi_center_y)**2)**0.5

            if dist > 15:   # tunable threshold
                defects.append(f"{roi_name}_SHIFTED")

    return list(set(defects))
