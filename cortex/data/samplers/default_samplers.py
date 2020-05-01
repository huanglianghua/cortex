import itertools
import torch
import math
from torch.utils.data.sampler import Sampler

import cortex.ops as ops


__all__ = ['InfiniteSampler', 'TrainSampler', 'TestSampler']


class InfiniteSampler(Sampler):

    def __init__(self, size, shuffle=True, seed=None):
        self._size = size
        self._shuffle = shuffle
        if seed is None:
            seed = ops.shared_random_seed()
        self._seed = int(seed)
        self._rank = ops.get_rank()
        self._world_size = ops.get_world_size()
    
    def __iter__(self):
        start, stop, step = self._rank, None, self._world_size
        yield from itertools.islice(
            self._infinite_indices(), start, stop, step)
    
    def _infinite_indices(self):
        g = torch.Generator()
        g.manual_seed(self._seed)
        while True:
            if self._shuffle:
                yield from torch.randperm(self._size, generator=g)
            else:
                yield from torch.arange(self._size)
    
    @property
    def length(self):
        # in replace of __len__, used for calculating the epoch size
        return self._size // self._world_size


TrainSampler = InfiniteSampler


class TestSampler(Sampler):

    def __init__(self, size):
        self._size = size
        self._rank = ops.get_rank()
        self._world_size = ops.get_world_size()

        local_size = int(math.ceil(self._size / self._world_size))
        start = local_size * self._rank
        stop = min(local_size * (self._rank + 1), self._size)
        self._local_inds = range(start, stop)
    
    def __iter__(self):
        yield from self._local_inds
    
    def __len__(self):
        return len(self._local_inds)
