import random
import numpy as np

def set_global_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    # torch.manual_seed(seed)  # if using PyTorch, etc.
    # ...add any others as needed