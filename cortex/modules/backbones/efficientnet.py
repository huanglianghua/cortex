import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
import re
import math
from collections import namedtuple
from functools import partial
from torch.utils.model_zoo import load_url


__all__ = [
    'EfficientNet', 'efficientnet_b0', 'efficientnet_b1',
    'efficientnet_b2', 'efficientnet_b3', 'efficientnet_b4',
    'efficientnet_b5', 'efficientnet_b6', 'efficientnet_b7']


cfgs = {
    # coefficients: width, depth, resolution, dropout
    'efficientnet-b0': (1.0, 1.0, 224, 0.2),
    'efficientnet-b1': (1.0, 1.1, 240, 0.2),
    'efficientnet-b2': (1.1, 1.2, 260, 0.3),
    'efficientnet-b3': (1.2, 1.4, 300, 0.3),
    'efficientnet-b4': (1.4, 1.8, 380, 0.4),
    'efficientnet-b5': (1.6, 2.2, 456, 0.4),
    'efficientnet-b6': (1.8, 2.6, 528, 0.5),
    'efficientnet-b7': (2.0, 3.1, 600, 0.5)}

model_urls = {
    'efficientnet-b0': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b0-355c32eb.pth',
    'efficientnet-b1': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b1-f1951068.pth',
    'efficientnet-b2': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b2-8bb594d6.pth',
    'efficientnet-b3': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b3-5fb5a3c3.pth',
    'efficientnet-b4': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b4-6ed6700e.pth',
    'efficientnet-b5': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b5-b6417697.pth',
    'efficientnet-b6': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b6-c76e70fd.pth',
    'efficientnet-b7': 'http://storage.googleapis.com/public-models/efficientnet/efficientnet-b7-dcc49843.pth'}

block_args = [
    # resolution, kernel_size, stride, expansion, in_channels,
    # out_channels, squeeze_ratio of SEModule
    'r1_k3_s1_e1_i32_o16_se0.25',
    'r2_k3_s2_e6_i16_o24_se0.25',
    'r2_k5_s2_e6_i24_o40_se0.25',
    'r3_k3_s2_e6_i40_o80_se0.25',
    'r3_k5_s1_e6_i80_o112_se0.25',
    'r4_k5_s2_e6_i112_o192_se0.25',
    'r1_k3_s1_e6_i192_o320_se0.25']

_BN_MOMENTUM = 1. - 0.99
_BN_EPSILON = 1e-3
_USE_SWISH = True
_DIVISOR = 8
_DROP_CONNECT = 0.2


def _decode_block_args(args_string):
    args = args_string.split('_')
    cfgs = {}
    for arg in args:
        splits = re.split(r'(\d.*)', arg)
        if len(splits) >= 2:
            key, val = splits[:2]
            cfgs[key] = val
    cfgs = dict(
        repeat=int(cfgs['r']),
        kernel_size=int(cfgs['k']),
        stride=int(cfgs['s']),
        expansion=int(cfgs['e']),
        in_channels=int(cfgs['i']),
        out_channels=int(cfgs['o']),
        se_ratio=float(cfgs['se']))
    return namedtuple('BlockArgs', cfgs.keys())(**cfgs)


def _make_divisible(val, divisor, min_ratio=0.9):
    new_val = max(divisor, int(val + divisor / 2)) // divisor * divisor
    # make sure that round down does not go down by more than 10%
    if new_val < min_ratio * val:
        new_val += divisor
    return new_val


def _drop_connect(x, p, training):
    if not training:
        return x
    keep_prob = 1. - p
    rand_tensor = torch.rand(
        (x.size(0), 1, 1, 1),
        dtype=x.dtype, device=x.device)
    return x * (rand_tensor + keep_prob).floor() / keep_prob


class _Swish(autograd.Function):

    @staticmethod
    def forward(ctx, i):
        result = i * i.sigmoid()
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = i.sigmoid()
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


swish = _Swish.apply


class Swish(nn.Module):

    def forward(self, x):
        return swish(x)


_activate = swish if _USE_SWISH else F.relu6
_Activate = Swish if _USE_SWISH else partial(nn.ReLU6, inplace=True)


class _Conv2d(nn.Conv2d):

    def forward(self, x):
        if self.stride[0] == 2:
            x = F.pad(x, [self.padding[0] - 1,
                          self.padding[1],
                          self.padding[0] - 1,
                          self.padding[1]])
            return F.conv2d(x, self.weight, self.bias, self.stride,
                            0, self.dilation, self.groups)
        else:
            return super(_Conv2d, self).forward(x)


class SEModule(nn.Module):

    def __init__(self, in_channels, squeeze_ratio):
        super(SEModule, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.reduce = _Conv2d(
            in_channels, int(in_channels * squeeze_ratio), 1)
        self.expand = _Conv2d(
            int(in_channels * squeeze_ratio), in_channels, 1)
    
    def forward(self, x):
        out = self.avgpool(x)
        out = _activate(self.reduce(out))
        out = torch.sigmoid(self.expand(out))
        return x * out


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, groups=1, activate=True):
        padding = (kernel_size - 1) // 2
        layers = [
            _Conv2d(in_channels, out_channels, kernel_size,
                    stride, padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels, momentum=_BN_MOMENTUM,
                           eps=_BN_EPSILON)]
        if activate:
            layers += [_Activate()]
        super(ConvBlock, self).__init__(*layers)


class MBConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride,
                 expansion, se_ratio=0.25):
        super(MBConvBlock, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        channels = int(round(in_channels * expansion))
        self.apply_residual = (self.stride == 1) and \
            (in_channels == out_channels)
        
        layers = []

        # pointwise convolution
        if expansion != 1:
            layers += [ConvBlock(in_channels, channels, 1)]
        
        # depthwise separable convolution
        layers += [ConvBlock(
            channels, channels, kernel_size,
            stride=stride, groups=channels)]
        
        # squeeze-excitation module
        if se_ratio is not None:
            se_ratio *= in_channels / channels
            layers += [SEModule(channels, squeeze_ratio=se_ratio)]
        
        # pointwise convolution
        layers += [ConvBlock(
            channels, out_channels, 1, activate=False)]
        
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x, drop_connect=None):
        # forward pass and dropout connect
        out = self.layers(x)
        if drop_connect:
            out = _drop_connect(
                out, p=drop_connect, training=self.training)
        
        # residual connection
        if self.apply_residual:
            return out + x
        else:
            return out


class EfficientNet(nn.Module):

    def __init__(self, block_args, coefficients, num_classes=None):
        super(EfficientNet, self).__init__()
        width_mul, depth_mul, resolution, dropout = coefficients
        self.width_mul = width_mul
        self.depth_mul = depth_mul
        self.input_size = resolution
        self.dropout = dropout

        # stem layers
        in_channels = 3
        out_channels = _make_divisible(32 * width_mul, _DIVISOR)
        self.conv1 = _Conv2d(
            3, out_channels, 3, 2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(
            out_channels, momentum=_BN_MOMENTUM, eps=_BN_EPSILON)
        self.activate = _Activate()

        # building blocks
        layers = []
        for args_string in block_args:
            # decode and update args based on width/depth multipliers
            args = _decode_block_args(args_string)
            args = args._replace(
                in_channels=_make_divisible(
                    args.in_channels * width_mul, _DIVISOR),
                out_channels=_make_divisible(
                    args.out_channels * width_mul, _DIVISOR),
                repeat=int(math.ceil(args.repeat * depth_mul)))
            
            # build layers by stacking MBConvBlocks
            layers += [MBConvBlock(
                args.in_channels, args.out_channels, args.kernel_size,
                args.stride, args.expansion, args.se_ratio)]
            if args.repeat > 1:
                for _ in range(1, args.repeat):
                    layers += [MBConvBlock(
                        args.out_channels, args.out_channels, args.kernel_size,
                        1, args.expansion, args.se_ratio)]
        self.layers = nn.ModuleList(layers)

        # head layers
        in_channels = args.out_channels
        out_channels = _make_divisible(1280 * width_mul, _DIVISOR)
        self.conv_head = _Conv2d(
            in_channels, out_channels, 1, bias=False)
        self.bn_head = nn.BatchNorm2d(
            out_channels, momentum=_BN_MOMENTUM, eps=_BN_EPSILON)
        self.activate_head = _Activate()

        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=self.dropout),
                nn.Linear(out_channels, num_classes))
        else:
            self.classifier = None

        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        # stem layers
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.activate(x)

        # MBConvBlock layers
        for i, layer in enumerate(self.layers):
            drop_connect = \
                _DROP_CONNECT * float(i) / len(self.layers)
            x = layer(x, drop_connect=drop_connect)
        
        # head layers
        x = self.activate_head(self.bn_head(self.conv_head(x)))

        # classification layer
        if self.classifier:
            x = self.classifier(x)

        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                # it's ok to use 'relu' as the nonlinearity since it
                # ensures that 'gain=math.sqrt(2)'
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def _efficientnet(arch, pretrained=False, **kwargs):
    model = EfficientNet(block_args, cfgs[arch], **kwargs)
    if pretrained:
        state_dict = load_url(model_urls[arch])
        model_state_dict = model.state_dict()
        for k1, k2 in zip(model_state_dict, state_dict):
            assert model_state_dict[k1].shape == state_dict[k2].shape
            model_state_dict[k1] = state_dict[k2]
        model.load_state_dict(model_state_dict)
    return model


def efficientnet_b0(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b0', pretrained, **kwargs)


def efficientnet_b1(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b1', pretrained, **kwargs)


def efficientnet_b2(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b2', pretrained, **kwargs)


def efficientnet_b3(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b3', pretrained, **kwargs)


def efficientnet_b4(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b4', pretrained, **kwargs)


def efficientnet_b5(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b5', pretrained, **kwargs)


def efficientnet_b6(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b6', pretrained, **kwargs)


def efficientnet_b7(pretrained=False, **kwargs):
    return _efficientnet('efficientnet-b7', pretrained, **kwargs)
