PCB_DATABASE = {
    "PCB_A": "backend/data/golden_pcb.jpg"
}

MODEL_PATH = "backend/models/best.pt"

# Detection stability thresholds
YOLO_CONF_THRESHOLD = 0.45
ALIGNMENT_MIN_SCORE = 0.30
SSIM_FAIL_THRESHOLD = 0.75

# ROI settings
ROI_EXPAND_RATIO = 1.2

# Stability control
ENABLE_DETERMINISTIC = True
RANDOM_SEED = 42
