import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url
from collections import OrderedDict


__all__ = ['ShuffleNetV2', 'shufflenetv2_x0_5', 'shufflenetv2_x1_0',
           'shufflenetv2_x1_5', 'shufflenetv2_x2_0']


model_urls = {
    'shufflenetv2_x0.5': 'https://download.pytorch.org/models/shufflenetv2_x0.5-f707e7126e.pth',
    'shufflenetv2_x1.0': 'https://download.pytorch.org/models/shufflenetv2_x1-5666bf0f80.pth',
    'shufflenetv2_x1.5': None,
    'shufflenetv2_x2.0': None}


def shuffle_channels(x, groups):
    n, c, h, w = x.size()
    channels_per_group = c // groups

    # reshape
    x = x.view(n, groups, channels_per_group, h, w)
    x = torch.transpose(x, 1, 2).contiguous()

    # flatten
    x = x.view(n, -1, h, w)

    return x


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride):
        super(InvertedResidual, self).__init__()
        if not (1 <= stride <= 3):
            raise ValueError('Illegal stride value')
        self.stride = stride

        branch_channels = out_channels // 2
        assert (self.stride != 1) or \
            (in_channels == branch_channels << 1)
        
        if self.stride > 1:
            self.branch1 = nn.Sequential(
                # depthwise separable convolution
                nn.Conv2d(in_channels, in_channels, 3, stride,
                          padding=1, bias=False, groups=in_channels),
                nn.BatchNorm2d(in_channels),
                # pointwise convolution
                nn.Conv2d(in_channels, branch_channels, 1, 1,
                          padding=0, bias=False),
                nn.BatchNorm2d(branch_channels),
                nn.ReLU(inplace=True))
        
        in_channels = branch_channels if stride == 1 else in_channels
        self.branch2 = nn.Sequential(
            # pointwise convolution
            nn.Conv2d(in_channels, branch_channels, 1, 1,
                      padding=0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True),
            # depthwise separable convolution
            nn.Conv2d(branch_channels, branch_channels, 3, stride,
                      padding=1, bias=False, groups=branch_channels),
            nn.BatchNorm2d(branch_channels),
            # pointwise convolution
            nn.Conv2d(branch_channels, branch_channels, 1, 1,
                      padding=0, bias=False),
            nn.BatchNorm2d(branch_channels),
            nn.ReLU(inplace=True))

    def forward(self, x):
        if self.stride == 1:
            x1, x2 = x.chunk(2, dim=1)
            out = torch.cat((x1, self.branch2(x2)), dim=1)
        else:
            out = torch.cat((self.branch1(x), self.branch2(x)), dim=1)
        out = shuffle_channels(out, 2)
        return out


class ShuffleNetV2(nn.Module):

    def __init__(self, blocks, block_channels, num_classes=None):
        super(ShuffleNetV2, self).__init__()
        assert len(blocks) == 3
        assert len(block_channels) == 5
        self._block_channels = block_channels

        in_channels = 3
        out_channels = self._block_channels[0]
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 2,
                      padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        in_channels = out_channels
        self.maxpool = nn.MaxPool2d(3, 2, padding=1)

        for i in range(len(blocks)):
            out_channels = self._block_channels[i + 1]
            layers = [InvertedResidual(in_channels, out_channels, 2)]
            for _ in range(1, blocks[i]):
                layers += [InvertedResidual(
                    out_channels, out_channels, 1)]
            setattr(self, 'layer{}'.format(i + 2),
                    nn.Sequential(*layers))
            in_channels = out_channels
        
        out_channels = self._block_channels[-1]
        self.conv5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, 1,
                      padding=0, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True))
        
        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(out_channels, num_classes))
        else:
            self.classifier = None
        
        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.conv5(x)
        if self.classifier is not None:
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


def shufflenetv2_x0_5(pretrained=False, **kwargs):
    model = ShuffleNetV2([4, 8, 4], [24, 48, 96, 192, 1024], **kwargs)
    if pretrained:
        arch = 'shufflenetv2_x0.5'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        state_dict = load_url(model_url, progress=True)
        state_dict = OrderedDict([
            (k.replace('stage', 'layer'), v)
            for k, v in state_dict.items()
            if not k.startswith('fc.')])
        model.load_state_dict(state_dict)
    return model


def shufflenetv2_x1_0(pretrained=False, **kwargs):
    model = ShuffleNetV2([4, 8, 4], [24, 116, 232, 464, 1024], **kwargs)
    if pretrained:
        arch = 'shufflenetv2_x1.0'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        state_dict = load_url(model_url, progress=True)
        state_dict = OrderedDict([
            (k.replace('stage', 'layer'), v)
            for k, v in state_dict.items()
            if not k.startswith('fc.')])
        model.load_state_dict(state_dict)
    return model


def shufflenetv2_x1_5(pretrained=False, **kwargs):
    model = ShuffleNetV2([4, 8, 4], [24, 176, 352, 704, 1024], **kwargs)
    if pretrained:
        arch = 'shufflenetv2_x1.5'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        state_dict = load_url(model_url, progress=True)
        state_dict = OrderedDict([
            (k.replace('stage', 'layer'), v)
            for k, v in state_dict.items()
            if not k.startswith('fc.')])
        model.load_state_dict(state_dict)
    return model


def shufflenetv2_x2_0(pretrained=False, **kwargs):
    model = ShuffleNetV2([4, 8, 4], [24, 244, 488, 976, 2048], **kwargs)
    if pretrained:
        arch = 'shufflenetv2_x2.0'
        model_url = model_urls[arch]
        if model_url is None:
            raise NotImplementedError(
                'pretrained {} is not supported as of now'.format(arch))
        state_dict = load_url(model_url, progress=True)
        state_dict = OrderedDict([
            (k.replace('stage', 'layer'), v)
            for k, v in state_dict.items()
            if not k.startswith('fc.')])
        model.load_state_dict(state_dict)
    return model
