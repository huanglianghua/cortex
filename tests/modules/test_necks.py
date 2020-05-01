import unittest
import torch

from cortex.modules.backbones import *
from cortex.modules.necks import *


class TestNecks(unittest.TestCase):

    def setUp(self):
        self.use_cpu = False
        self.cuda = False if self.use_cpu else torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.input = torch.rand((2, 3, 512, 512)).to(self.device)

    def test_fpn(self):
        nets = {
            'resnet18': resnet18(pretrained=True, out_layers=[
                'layer1', 'layer2', 'layer3', 'layer4']),
            'resnet50': resnet50(pretrained=True, out_layers=[
                'layer1', 'layer2', 'layer3', 'layer4']),
            'resnet101': resnet101(pretrained=True, out_layers=[
                'layer1', 'layer2', 'layer3'])}
        fpns = [
            FPN(in_channels=[64, 128, 256, 512], out_channels=256),
            FPN(in_channels=[256, 512, 1024, 2048], out_channels=256),
            FPN(in_channels=[256, 512, 1024], out_channels=256)]
        
        for i, name in enumerate(nets):
            net = nets[name].to(self.device)
            fpn = fpns[i].to(self.device)

            # random input tensor
            if hasattr(net, 'input_size'):
                size = net.input_size
                input = torch.rand(2, 3, size, size).to(self.device)
            else:
                input = self.input

            # forward pass
            outs = net(self.input)
            if isinstance(outs, dict):
                outs = list(outs.values())
            fpn_outs = fpn(outs)

            # print inference information
            print('{}_FPN:'.format(name))
            for layer, out, fpn_out in zip(
                net.out_layers, outs, fpn_outs):
                print('  [{}] input: {} output: {}'.format(
                    layer,
                    tuple(out.shape),
                    tuple(fpn_out.shape)))


if __name__ == '__main__':
    unittest.main()
