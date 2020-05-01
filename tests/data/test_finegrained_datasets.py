import unittest
import random
from PIL import Image

import cortex.data as data


class TestFineGrainedDatasets(unittest.TestCase):

    def test_cub200(self):
        self._test_dataset(data.CUB200)
    
    def test_cars196(self):
        self._test_dataset(data.Cars196)
    
    def test_stanford_online_products(self):
        self._test_dataset(data.StanfordOnlineProducts)

    def _test_dataset(self, dataset_class, root_dir=None):
        # check non-overlapping between train and test classes
        train_data = dataset_class(
            root_dir, subset='train', split_method='classwise')
        test_data = dataset_class(
            root_dir, subset='test', split_method='classwise')
        self.assertEqual(
            len(set(train_data.CLASSES) & set(test_data.CLASSES)), 0)
        
        # sanity check
        for subset in [train_data, test_data]:
            for _ in range(100):
                item = random.choice(subset)
                self.assertEqual(len(item), 2)
                self.assertTrue(isinstance(item[0], Image.Image))
                self.assertLess(item[1], len(subset.CLASSES))


if __name__ == '__main__':
    unittest.main()
