import unittest
import numpy as np

import cortex.data as data


class TestDefaultSamplers(unittest.TestCase):

    def setUp(self):
        self.world_size = 4
        self.dataset_size = 1000

    def test_train_sampler(self):
        sampler = data.TrainSampler(
            size=self.dataset_size, shuffle=False)
        sampler._world_size = self.world_size
        for rank in range(self.world_size):
            sampler._rank = rank
            iterator = iter(sampler)
            inds = []
            for _ in range(100000):
                inds.append(next(iterator))
            inds = np.array(inds)
            self.assertTrue(
                np.all((inds - rank) % self.world_size == 0))
    
    def test_inference_sampler(self):
        # world_size=1, rank=0
        sampler = data.TestSampler(size=self.dataset_size)
        inds = [i for i in sampler]
        self.assertTrue(np.all(inds == np.arange(self.dataset_size)))


if __name__ == '__main__':
    unittest.main()
