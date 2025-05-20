import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from LibMTL.metrics import AbsMetric
from LibMTL.loss import AbsLoss

class QM9Metric(AbsMetric):
    r"""Calculate the Mean Absolute Error (MAE).
    """
    def __init__(self, std, scale=1):
        super(QM9Metric, self).__init__()

        self.std = std
        self.scale = scale
        
    def update_fun(self, pred, gt):
        r"""
        """
        abs_err = torch.abs(pred * (self.std).to(pred.device) - gt * (self.std).to(pred.device)).view(pred.size()[0], -1).sum(-1)
        self.record.append(abs_err.cpu().numpy())
        
    def score_fun(self):
        r"""
        """
        records = np.concatenate(self.record)
        return [records.mean()*self.scale]
    
from torch.utils.data import Sampler
class PeriodicSquareRootSampler(Sampler):
    r"""Batch sampler with varying batch sizes. The batch size changes
        periodically based on the iteration number.
    """
    def __init__(self, sampler, n=None, q=None, drop_last=False):
        self.sampler = sampler
        self.n = n
        self.q = q #math.ceil(math.sqrt(self.n))
        self.step=0
        self.drop_last = drop_last

    def __len__(self):
        if self.drop_last:
            pass
        else:
            y = len(self.sampler)
            periods=math.floor(y / ((2*self.n)-self.q))
            chunks=y-(periods*((2*self.n) - self.q))
            num_batches = periods*self.q
            if chunks > 0:
                num_batches += 1
                if (chunks - self.n) > 0:
                    num_batches += math.ceil((chunks - self.n)/self.q)
            return num_batches

    def __iter__(self):
        self.batch_size = self.n if (self.step % self.q == 0) else self.q
        if self.drop_last:
            sampler_iter = iter(self.sampler)
            while True:
                try:
                    batch = [next(sampler_iter) for _ in range(self.batch_size)]
                    yield batch
                    self.step += 1
                    self.batch_size = self.n if (self.step % self.q == 0) else self.q
                except StopIteration:
                    break
        else:
            batch = [0] * self.batch_size
            idx_in_batch = 0
            for idx in self.sampler:
                batch[idx_in_batch] = idx
                idx_in_batch += 1
                if idx_in_batch == self.batch_size:
                    yield batch
                    self.step += 1
                    self.batch_size = self.n if (self.step % self.q == 0) else self.q
                    idx_in_batch = 0
                    batch = [0] * self.batch_size
            if idx_in_batch > 0:
                yield batch[:idx_in_batch]
                self.step += 1
                self.batch_size = self.n if (self.step % self.q == 0) else self.q