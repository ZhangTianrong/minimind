def seed_everything(seed) -> int:
    import os
    os.environ['PYTHONHASHSEED'] = str(seed)
    
    import random
    random.seed(seed)

    
    import numpy as np
    max_seed_value = np.iinfo(np.uint32).max
    min_seed_value = np.iinfo(np.uint32).min
    assert (min_seed_value <= seed <= max_seed_value), f"Seed value {seed} not in bounds. Numpy accepts seeds from {min_seed_value} to {max_seed_value} only."
    np.random.seed(seed)
    
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True
    
    from transformers import set_seed
    set_seed(seed)

    return