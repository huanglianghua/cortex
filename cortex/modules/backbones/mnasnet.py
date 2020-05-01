import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from collections import OrderedDict


__all__ = ['MNasNet', 'mnasnet0_5', 'mnasnet0_75', 'mnasnet1_0',
           'mnasnet1_3']


model_urls = {
    "mnasnet0_5":
    "https://download.pytorch.org/models/mnasnet0.5_top1_67.592-7c6cb539b9.pth",
    "mnasnet0_75": None,
    "mnasnet1_0":
    "https://download.pytorch.org/models/mnasnet1.0_top1_73.512-f206786ef8.pth",
    "mnasnet1_3": None}

_BN_MOMENTUM = 1. - 0.9997


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size,
                 stride, expansion, bn_momentum=0.1):
        super(InvertedResidual, self).__init__()
        assert stride in [1, 2]
        assert kernel_size in [3, 5]
        channels = in_channels * expansion
        self.apply_residual = (stride == 1) and \
            (in_channels == out_channels)
        self.layers = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # depthwise separable convolution
            nn.Conv2d(channels, channels, kernel_size, stride,
                      padding=kernel_size // 2, groups=channels,
                      bias=False),
            nn.BatchNorm2d(channels, momentum=bn_momentum),
            nn.ReLU(inplace=True),
            # pointwise convolution
            nn.Conv2d(channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels, momentum=bn_momentum))
    
    def forward(self, x):
        if self.apply_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


def _make_divisible(val, divisor, min_ratio=0.9):
    new_val = max(divisor, int(val + divisor / 2)) // divisor * divisor
    # make sure that round down does not go down by more than 10%
    if new_val < min_ratio * val:
        new_val += divisor
    return new_val


def _scale_depths(depths, alpha):
    return [_make_divisible(depth * alpha, 8) for depth in depths]


def _stack(in_channels, out_channels, kernel_size, stride,
           expansion, repeats, bn_momentum):
    layers = [InvertedResidual(
        in_channels, out_channels, kernel_size, stride,
        expansion, bn_momentum=bn_momentum)]
    for _ in range(1, repeats):
        layers += [InvertedResidual(
            out_channels, out_channels, kernel_size, 1,
            expansion, bn_momentum=bn_momentum)]
    return nn.Sequential(*layers)


class MNasNet(nn.Module):

    def __init__(self, alpha, num_classes=None, dropout=0.2):
        super(MNasNet, self).__init__()
        depths = _scale_depths([24, 40, 80, 96, 192, 320], alpha)
        self.layers = nn.Sequential(
            # first layer
            nn.Conv2d(3, 32, 3, 2, padding=1, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            # depthwise separable convolution
            nn.Conv2d(32, 32, 3, 1, padding=1, groups=32, bias=False),
            nn.BatchNorm2d(32, momentum=_BN_MOMENTUM),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 16, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(16, momentum=_BN_MOMENTUM),
            # MNasNet blocks
            _stack(16, depths[0], 3, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[0], depths[1], 5, 2, 3, 3, _BN_MOMENTUM),
            _stack(depths[1], depths[2], 5, 2, 6, 3, _BN_MOMENTUM),
            _stack(depths[2], depths[3], 3, 1, 6, 2, _BN_MOMENTUM),
            _stack(depths[3], depths[4], 5, 2, 6, 4, _BN_MOMENTUM),
            _stack(depths[4], depths[5], 3, 1, 6, 1, _BN_MOMENTUM),
            # last layer
            nn.Conv2d(depths[5], 1280, 1, 1, padding=0, bias=False),
            nn.BatchNorm2d(1280, momentum=_BN_MOMENTUM))
        
        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=dropout, inplace=True),
                nn.Linear(1280, num_classes))
        else:
            self.classifier = None

        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        x = self.layers(x)
        if self.classifier:
            x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.01)
                nn.init.zeros_(m.bias)


def mnasnet0_5(pretrained=False, **kwargs):
    model = MNasNet(0.5, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['mnasnet0_5'], progress=True),
            strict=False)
    return model


def mnasnet0_75(pretrained=False, **kwargs):
    model = MNasNet(0.75, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['mnasnet0_75'], progress=True),
            strict=False)
    return model


def mnasnet1_0(pretrained=False, **kwargs):
    model = MNasNet(1.0, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['mnasnet1_0'], progress=True),
            strict=False)
    return model


def mnasnet1_3(pretrained=False, **kwargs):
    model = MNasNet(1.3, **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['mnasnet1_3'], progress=True),
            strict=False)
    return model
