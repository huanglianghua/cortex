import unittest
import os
import torch
import numpy as np

import cortex.data as data
import cortex.ops as ops


def _mper_class_sampler(labels, batch_size, num_instances):
    rank = ops.get_rank()
    sampler = data.MPerClassSampler(labels, batch_size, num_instances)
    iterator = iter(sampler)
    batch_inds = []
    for _ in range(batch_size):
        batch_inds.append(next(iterator))
    batch_labels = [labels[i] for i in batch_inds]
    u_labels, u_counts = np.unique(batch_labels, return_counts=True)
    print('rank: {} sampler_seed: {} labels: {} counts: {}'.format(
        rank, sampler.seed, u_labels, u_counts))
    assert np.all(u_counts == num_instances)


class TestMetricLearningSamplers(unittest.TestCase):

    def setUp(self):
        self.dataset = data.CUB200(subset='full')
        self.batch_size = 128
        self.num_instances = 8

        # distributed settings
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        self.gpus_per_machine = torch.cuda.device_count()
    
    def test_mper_class_sampler(self):
        # test MPerClassSampler on single GPU
        sampler = data.MPerClassSampler(
            self.dataset.labels, self.batch_size, self.num_instances)
        iterator = iter(sampler)
        batch_inds = []
        for _ in range(self.batch_size):
            batch_inds.append(next(iterator))
        labels = [self.dataset.labels[i] for i in batch_inds]
        u_labels, u_counts = np.unique(labels, return_counts=True)
        self.assertTrue(np.all(u_counts == self.num_instances))

        # test MPerClassSampler on multiple GPUs (distributed sampling)
        ops.launch(
            _mper_class_sampler,
            args=(self.dataset.labels,
                  self.batch_size,
                  self.num_instances),
            gpus_per_machine=self.gpus_per_machine)


if __name__ == '__main__':
    unittest.main()
