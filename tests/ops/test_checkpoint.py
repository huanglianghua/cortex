import unittest
import torch
import torch.nn as nn
import torch.optim as optim
import os

import cortex.ops as ops


class TestCheckpoint(unittest.TestCase):

    def test_checkpoint(self):
        model = nn.Sequential(
            nn.Conv2d(3, 3, 3),
            nn.BatchNorm2d(3),
            nn.ReLU(inplace=True))
        criterions = [nn.CrossEntropyLoss(), nn.SmoothL1Loss()]
        optimizers = [
            optim.SGD(model.parameters(), lr=0.001),
            optim.Adam(model.parameters(), lr=0.1)]
        filename = './tmp.pth'

        # set parameters to random and save checkpoint
        nn.init.uniform_(model[0].weight)
        nn.init.normal_(model[0].bias)
        nn.init.normal_(model[1].weight)
        saved_state = {
            k: v.clone() for k, v in model.state_dict().items()}
        ops.save_checkpoint(filename, model, criterions, optimizers)

        # set parameters to zeros
        nn.init.zeros_(model[0].weight)
        nn.init.zeros_(model[0].bias)
        nn.init.zeros_(model[1].weight)
        self.assertEqual(model[0].weight.abs().mean(), 0)

        # load checkpoint and check the parameters
        ops.load_checkpoint(filename, model, criterions, optimizers)
        for k, v in model.state_dict().items():
            v0 = saved_state[k]
            self.assertTrue(torch.all(v == v0))

        # remove temporal checkpoint file
        os.remove(filename)


if __name__ == '__main__':
    unittest.main()
