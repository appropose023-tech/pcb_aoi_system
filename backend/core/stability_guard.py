import numpy as np
import random
import torch
from config import RANDOM_SEED

def set_deterministic():
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    torch.manual_seed(RANDOM_SEED)
    torch.cuda.manual_seed_all(RANDOM_SEED)

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
