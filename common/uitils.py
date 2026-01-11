#工具类文件




import numpy as np
import torch
import random
import os

#使用统一的随机数种子
def set_seed(seed):
    """Set the random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  #cpu
    #torch.cuda.manual_seed(seed)  #gpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


#

                                        