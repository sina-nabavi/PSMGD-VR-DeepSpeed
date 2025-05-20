from torch.utils.data import Sampler
import math

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