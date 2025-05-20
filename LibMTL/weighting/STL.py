import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from LibMTL.weighting.abstract_weighting import AbsWeighting

class STL(AbsWeighting):
    r"""Single Task Learning (STL).

    Assigns a weight of 1 to the main task (`main_task`) and 0 to all others.

    Args:
        main_task (int): The index of the main task (0-indexed).
    """
    def __init__(self):
        super(STL, self).__init__()
        
    def backward(self, losses, prev_train_losses=None, **kwargs):
        main_task = kwargs['main_task']
        weights = torch.zeros_like(losses).to(self.device)
        weights[main_task] = 1.0
        loss = torch.mul(losses, weights).sum()
        loss.backward()
        return weights.cpu().numpy()