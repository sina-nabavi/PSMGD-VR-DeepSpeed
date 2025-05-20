import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class EW_DS(AbsWeighting):
    r"""Equal Weighting (EW) with deepspeed.

    The loss weight for each task is always ``1 / T`` in every iteration, where ``T`` denotes the number of tasks.

    """
    def __init__(self):
        super(EW_DS, self).__init__()
        
    def backward(self, losses, backward, **kwargs):
        loss = torch.mul(losses, torch.ones_like(losses).to(self.device)).sum()
        backward(loss)
        return np.ones(self.task_num)