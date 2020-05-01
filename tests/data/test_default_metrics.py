import unittest
import torch
import numpy as np

import cortex.data as data


class TestDefaultMetrics(unittest.TestCase):

    def test_default_metrics(self):
        avg_metric = data.Average()
        single_metric = data.Average()
        for _ in range(100000):
            output = {
                'm1': np.random.randn(),
                'm2': torch.randn(1),
                'm3': 5}
            _ = avg_metric.update(output)
            _ = single_metric.update(output['m2'])
        metrics = avg_metric.compute()
        self.assertAlmostEqual(float(metrics['m1']), 0, places=1)
        self.assertAlmostEqual(float(metrics['m2']), 0, places=1)
        self.assertEqual(float(metrics['m3']), 5)
        metrics = single_metric.compute()
        self.assertAlmostEqual(float(metrics['loss']), 0, places=1)


if __name__ == '__main__':
    unittest.main()
