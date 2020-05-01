import unittest
import torch
import time

from cortex.modules.backbones import *


class TestBackbones(unittest.TestCase):

    def setUp(self):
        self.cuda = True and torch.cuda.is_available()
        self.device = torch.device('cuda:0' if self.cuda else 'cpu')
        self.input = torch.randn((2, 3, 512, 512)).to(self.device)
    
    def test_alexnet(self):
        nets = {
            'AlexNet': AlexNet(out_layers='conv4'),
            'AlexNetV1': AlexNetV1(out_layers='conv5'),
            'AlexNetV2': AlexNetV2(out_layers=['conv4', 'conv5']),
            'BigAlexNet': BigAlexNet(out_layers=['conv5']),
            'alexnet': alexnet(pretrained=True, out_layers='conv5')}
        self._check_nets(nets)
    
    def test_vggnet(self):
        nets = {
            'vgg11': vgg11(pretrained=True),
            'vgg13_bn': vgg13_bn(pretrained=True),
            'vgg16': vgg16(pretrained=True),
            'vgg19_bn': vgg19_bn(pretrained=True),
            'ssd300_vgg': ssd300_vgg(pretrained=True),
            'ssd512_vgg': ssd512_vgg(pretrained=True)}
        self._check_nets(nets)
    
    def test_resnet(self):
        nets = {
            'resnet18': resnet18(pretrained=True, out_layers='layer4'),
            'resnet34': resnet34(pretrained=True, out_layers=['layer3', 'layer4']),
            'resnet50': resnet50(pretrained=True, out_layers=['layer3', 'layer4']),
            'resnet101': resnet101(pretrained=True, out_layers=['layer3', 'layer4']),
            'resnet152': resnet152(pretrained=True, out_layers=['layer3', 'layer4']),
            'resnext50_32x4d': resnext50_32x4d(pretrained=True, out_layers=['layer3', 'layer4']),
            'resnext101_32x8d': resnext101_32x8d(pretrained=True, out_layers=['layer3', 'layer4']),
            'wide_resnet50_2': wide_resnet50_2(pretrained=True, out_layers=['layer3', 'layer4']),
            'wide_resnet101_2': wide_resnet101_2(pretrained=True, out_layers=['layer1', 'layer2', 'layer3', 'layer4'])}
        self._check_nets(nets)
    
    def test_mobilenet(self):
        nets = {'mobilenet_v2': mobilenet_v2(pretrained=True)}
        self._check_nets(nets)
    
    def test_squeezenet(self):
        nets = {
            'squeezenet1_0': squeezenet1_0(pretrained=True),
            'squeezenet1_1': squeezenet1_1(pretrained=True)}
        self._check_nets(nets)
    
    def test_shufflenet(self):
        nets = {
            'shufflenetv2_x0_5': shufflenetv2_x0_5(pretrained=True),
            'shufflenetv2_x1_0': shufflenetv2_x1_0(pretrained=True),
            'shufflenetv2_x1_5': shufflenetv2_x1_5(pretrained=False),
            'shufflenetv2_x2_0': shufflenetv2_x2_0(pretrained=False)}
        self._check_nets(nets)
    
    def test_mnasnet(self):
        nets = {
            'mnasnet0_5': mnasnet0_5(pretrained=True),
            'mnasnet0_75': mnasnet0_75(pretrained=False),
            'mnasnet1_0': mnasnet1_0(pretrained=True),
            'mnasnet1_3': mnasnet1_3(pretrained=False)}
        self._check_nets(nets)
    
    def test_efficientnet(self):
        nets = {
            'efficientnet_b0': efficientnet_b0(pretrained=True),
            'efficientnet_b1': efficientnet_b1(pretrained=True),
            'efficientnet_b2': efficientnet_b2(pretrained=True),
            'efficientnet_b3': efficientnet_b3(pretrained=True),
            'efficientnet_b4': efficientnet_b4(pretrained=True),
            'efficientnet_b5': efficientnet_b5(pretrained=True),
            'efficientnet_b6': efficientnet_b6(pretrained=True),
            'efficientnet_b7': efficientnet_b7(pretrained=True)}
        self._check_nets(nets)
    
    def test_drnet(self):
        nets = {'drnet': DRNet()}
        self._check_nets(nets)
    
    def test_bninception(self):
        nets = {'bninception': bninception(pretrained=True)}
        self._check_nets(nets)
    
    def _check_nets(self, nets):
        if self.cuda:
            print('Runtime on GPU:')
            # warmup GPU
            _ = resnet50().to(self.device)(self.input)

        for name, net in nets.items():
            net = net.to(self.device)

            # random input tensor
            if hasattr(net, 'input_size'):
                size = net.input_size
                input = torch.randn((2, 3, size, size)).to(self.device)
            else:
                input = self.input

            # forward pass
            start = time.perf_counter()
            output = net(input)
            stop = time.perf_counter()

            # extract the last layer output
            if isinstance(output, dict):
                key = list(output.keys())[-1]
                output = output[key]
            elif isinstance(output, list):
                output = output[-1]
            
            # print inference information
            print('[{}] input: {} output: {} speed: {:.1f} fps'.format(
                name,
                tuple(input.shape),
                tuple(output.shape),
                1. / (stop - start)))
            
            # clear variables
            del net, output


if __name__ == '__main__':
    unittest.main()
