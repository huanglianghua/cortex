import unittest
import torch

from cortex.modules.submodules import *


class TestSubmodules(unittest.TestCase):

    def setUp(self):
        cuda = torch.cuda.is_available()
        self.device = torch.device('cuda:0' if cuda else 'cpu')
        self.in_channels = 64
        self.input = torch.randn(
            2, self.in_channels, 32, 32).to(self.device)

    def test_semodule(self):
        for reduction in [1, 2, 4, 8, 16, 32]:
            net = SEModule(self.in_channels, reduction).to(self.device)
            out = net(self.input)
            self.assertEqual(tuple(self.input.shape), tuple(out.shape))
    
    def test_cbam(self):
        for reduction in [1, 2, 4, 8, 16, 32]:
            net = CBAM(self.in_channels, reduction).to(self.device)
            out = net(self.input)
            self.assertEqual(tuple(self.input.shape), tuple(out.shape))
    
    def test_frn(self):
        # fixed eps
        net = FilterResponseNorm2d(
            self.in_channels, 1e-6, False).to(self.device)
        out = net(self.input)
        self.assertEqual(tuple(self.input.shape), tuple(out.shape))

        # learnable eps
        net = FilterResponseNorm2d(
            self.in_channels, 1e-4, True).to(self.device)
        out = net(self.input)
        self.assertEqual(tuple(self.input.shape), tuple(out.shape))


if __name__ == '__main__':
    unittest.main()
