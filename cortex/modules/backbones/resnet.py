import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from collections import OrderedDict


__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


model_urls = {
    'resnet18': 'https://download.pytorch.org/models/resnet18-5c106cde.pth',
    'resnet34': 'https://download.pytorch.org/models/resnet34-333f7ec4.pth',
    'resnet50': 'https://download.pytorch.org/models/resnet50-19c8e357.pth',
    'resnet101': 'https://download.pytorch.org/models/resnet101-5d3b4d8f.pth',
    'resnet152': 'https://download.pytorch.org/models/resnet152-b121ed2d.pth',
    'resnext50_32x4d': 'https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth',
    'resnext101_32x8d': 'https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth',
    'wide_resnet50_2': 'https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth',
    'wide_resnet101_2': 'https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth'}


def conv3x3(in_channels, out_channels, stride=1, groups=1, dilation=1):
    return nn.Conv2d(
        in_channels, out_channels, 3, stride,
        padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 caffe_style=False):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError(
                'BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError(
                'Dilation > 1 not supported in BasicBlock')
        # both self.conv1 and self.downsample layers downsample the
        # input when stride != 1
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = norm_layer(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = norm_layer(channels)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, channels, stride=1, downsample=None,
                 groups=1, base_width=64, dilation=1, norm_layer=None,
                 caffe_style=False):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(channels * (base_width / 64.)) * groups
        if caffe_style:
            # both self.conv1 and self.downsample layers downsample the
            # input when stride != 1
            stride1 = stride
            stride2 = 1
        else:
            # both self.conv2 and self.downsample layers downsample the
            # input when stride != 1
            stride1 = 1
            stride2 = stride
        self.conv1 = conv1x1(in_channels, width, stride1)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride2, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, channels * self.expansion)
        self.bn3 = norm_layer(channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=None,
                 zero_init_residual=False, groups=1, width_per_group=64,
                 strides=[1, 2, 2, 2], dilations=[1, 1, 1, 1],
                 norm_layer=None, caffe_style=False, out_layers='layer4'):
        super(ResNet, self).__init__()
        if num_classes is not None and out_layers != 'layer4':
            raise ValueError(
                'Expected out_layers="layer4" when num_classes is '
                'specified, but got out_layers={}'.format(out_layers))
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self._caffe_style = caffe_style
        self.strides = strides
        self.dilations = dilations
        self.out_layers = out_layers

        self.in_channels = 64
        self.groups = groups
        self.base_width = width_per_group
        # build the first layer
        self.conv1 = nn.Conv2d(
            3, self.in_channels, 7, 2, padding=3, bias=False)
        self.bn1 = norm_layer(self.in_channels)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)
        # build layer 1~layer 4
        self.layer1 = self._make_layer(
            block, 64, layers[0], stride=strides[0],
            dilation=dilations[0], previous_dilation=1)
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=strides[1],
            dilation=dilations[1], previous_dilation=dilations[0])
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=strides[2],
            dilation=dilations[2], previous_dilation=dilations[1])
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=strides[3],
            dilation=dilations[3], previous_dilation=dilations[2])

        # build classifier if num_classes is not None
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(512 * block.expansion, num_classes))
        else:
            self.classifier = None

        # initialize weights
        self.init_weights(zero_init_residual)

    def _make_layer(self, block, channels, blocks, stride=1, dilation=1,
                    previous_dilation=1):
        norm_layer = self._norm_layer
        caffe_style = self._caffe_style
        downsample = None
        if stride != 1 or self.in_channels != channels * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.in_channels,
                        channels * block.expansion,
                        stride),
                norm_layer(channels * block.expansion))

        layers = []
        layers.append(block(
            self.in_channels, channels, stride, downsample, self.groups,
            self.base_width, previous_dilation, norm_layer, caffe_style))
        self.in_channels = channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(
                self.in_channels, channels, groups=self.groups,
                base_width=self.base_width, dilation=dilation,
                norm_layer=norm_layer, caffe_style=caffe_style))

        return nn.Sequential(*layers)

    def forward(self, x, out_layers=None):
        if out_layers is None:
            out_layers = self.out_layers

        if self.classifier is not None and out_layers != 'layer4':
            raise ValueError(
                'Expected out_layers="layer4" when num_classes is '
                'specified, but got out_layers={}'.format(out_layers))

        # check if returns single-layer output
        single_out = isinstance(out_layers, str)
        out = OrderedDict()

        # run the first layer
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        # run layer 1~layer 4
        for layer in ['layer1', 'layer2', 'layer3', 'layer4']:
            x = getattr(self, layer)(x)
            if self._add_and_check(layer, x, out_layers, out):
                out = out[out_layers] if single_out else out
                break

        return self.classifier(out) if self.classifier else out

    def _add_and_check(self, name, x, out_layers, out):
        if isinstance(out_layers, str):
            out_layers = [out_layers]
        assert all([l in ['layer1', 'layer2', 'layer3', 'layer4']
                    for l in out_layers])
        if name in out_layers:
            out[name] = x
        return len(out_layers) == len(out)

    def init_weights(self, zero_init_residual=False):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # zero-initialize the last BN in each residual branch
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)
                elif isinstance(m, Bottleneck):
                    nn.init.zeros_(m.bn3.weight)


def _load_pretrained(name, model):
    assert name in model_urls
    key_dict = {'fc.': 'classifier.2.'}
    src_state = load_url(model_urls[name], progress=True)
    dst_state = type(src_state)()
    for k, v in src_state.items():
        for k1, k2 in key_dict.items():
            k = k.replace(k1, k2)
        dst_state[k] = v
    return model.load_state_dict(dst_state, strict=False)


def resnet18(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)
    if pretrained:
        _load_pretrained('resnet18', model)
    return model


def resnet34(pretrained=False, **kwargs):
    model = ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnet34', model)
    return model


def resnet50(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnet50', model)
    return model


def resnet101(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnet101', model)
    return model


def resnet152(pretrained=False, **kwargs):
    model = ResNet(Bottleneck, [3, 8, 36, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnet152', model)
    return model


def resnext50_32x4d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnext50_32x4d', model)
    return model


def resnext101_32x8d(pretrained=False, **kwargs):
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained('resnext101_32x8d', model)
    return model


def wide_resnet50_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)
    if pretrained:
        _load_pretrained('wide_resnet50_2', model)
    return model


def wide_resnet101_2(pretrained=False, **kwargs):
    kwargs['width_per_group'] = 64 * 2
    model = ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)
    if pretrained:
        _load_pretrained('wide_resnet101_2', model)
    return model
