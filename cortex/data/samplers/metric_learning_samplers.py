import random
import copy
from torch.utils.data.sampler import Sampler
from collections import defaultdict

import cortex.ops as ops


__all__ = ['MPerClassSampler']


class MPerClassSampler(Sampler):

    def __init__(self, labels, batch_size, num_instances, seed=None):
        assert batch_size % num_instances == 0
        self.labels = labels
        self.batch_size = batch_size
        self.num_instances = num_instances
        self.num_labels = batch_size // num_instances
        if seed is None:
            seed = ops.shared_random_seed()
        self.seed = seed
        self.rank = ops.get_rank()
        self.world_size = ops.get_world_size()

        # collect instance IDs for each label
        self.label_dict = defaultdict(list)
        for i, label in enumerate(labels):
            self.label_dict[label].append(i)
        self.labels = list(self.label_dict.keys())

        # estimate sample number per epoch
        divisor = self.world_size * batch_size
        total_size = len(labels) // divisor * divisor
        self._size = total_size // self.world_size
    
    def __iter__(self):
        rng = random.Random(self.seed)
        # infinite sample index generator
        while True:
            yield from self._epoch_indices(rng)
    
    def _epoch_indices(self, rng):
        # collect batch indices for each label
        batch_dict = defaultdict(list)
        for label in self.labels:
            # collect all dataset indices for this label
            inds = copy.deepcopy(self.label_dict[label])
            if len(inds) < self.num_instances:
                inds = rng.choices(inds, k=self.num_instances)
            rng.shuffle(inds)

            # batchify the indices
            batch_inds = []
            for ind in inds:
                batch_inds.append(ind)
                if len(batch_inds) == self.num_instances:
                    batch_dict[label].append(batch_inds)
                    batch_inds = []
        
        # flatten batch indices of all labels
        flatten_inds = []
        available_labels = copy.deepcopy(self.labels)
        while len(available_labels) >= self.num_labels:
            selected_labels = rng.sample(
                available_labels, self.num_labels)
            for label in selected_labels:
                batch_inds = batch_dict[label].pop(0)
                flatten_inds.extend(batch_inds)
                if len(batch_dict[label]) == 0:
                    available_labels.remove(label)
        
        # drop last few samples if necessary to make size divisible
        divisor = self.world_size * self.batch_size
        num_samples = len(flatten_inds) // divisor * divisor
        flatten_inds = flatten_inds[:num_samples]

        # subsample indices for this rank
        samples_per_rank = num_samples // self.world_size
        start = samples_per_rank * self.rank
        stop = samples_per_rank * (self.rank + 1)
        rank_inds = flatten_inds[start:stop]
        self._size = samples_per_rank

        yield from rank_inds
    
    @property
    def length(self):
        # in replace of __len__, used for calculating the epoch size
        return self._size
