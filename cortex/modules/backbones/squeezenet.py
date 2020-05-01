import torch
import torch.nn as nn
from torch.utils.model_zoo import load_url


__all__ = ['SqueezeNet', 'squeezenet1_0', 'squeezenet1_1']


model_urls = {
    'squeezenet1_0': 'https://download.pytorch.org/models/squeezenet1_0-a815701f.pth',
    'squeezenet1_1': 'https://download.pytorch.org/models/squeezenet1_1-f364aa15.pth'}


class Fire(nn.Module):

    def __init__(self, in_channels, squeeze_channels,
                 expand1x1_channels, expand3x3_channels):
        super(Fire, self).__init__()
        self.in_channels = in_channels
        self.squeeze = nn.Conv2d(in_channels, squeeze_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.expand1x1 = nn.Conv2d(
            squeeze_channels, expand1x1_channels, 1)
        self.expand3x3 = nn.Conv2d(
            squeeze_channels, expand3x3_channels, 3, padding=1)
    
    def forward(self, x):
        x = self.relu(self.squeeze(x))
        return torch.cat([
            self.relu(self.expand1x1(x)),
            self.relu(self.expand3x3(x))], dim=1)


class SqueezeNet(nn.Module):

    def __init__(self, version='1_0', num_classes=None):
        super(SqueezeNet, self).__init__()
        if version == '1_0':
            self.features = nn.Sequential(
                nn.Conv2d(3, 96, 7, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(96, 16, 64, 64),
                Fire(128, 16, 64, 64),
                Fire(128, 32, 128, 128),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(256, 32, 128, 128),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(512, 64, 256, 256))
        elif version == '1_1':
            self.features = nn.Sequential(
                nn.Conv2d(3, 64, 3, 2),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(64, 16, 64, 64),
                Fire(128, 16, 64, 64),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(128, 32, 128, 128),
                Fire(256, 32, 128, 128),
                nn.MaxPool2d(3, 2, ceil_mode=True),
                Fire(256, 48, 192, 192),
                Fire(384, 48, 192, 192),
                Fire(384, 64, 256, 256),
                Fire(512, 64, 256, 256))
        else:
            raise ValueError(
                'Unsupported SqueezeNet version {version}:'
                '1_0 or 1_1 expected'.format(version=version))
        
        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Conv2d(512, num_classes, 1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten())
        else:
            self.classifier = None
        
        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        x = self.features(x)
        if self.classifier:
            x = self.classifier(x)
        return x
    
    def init_weights(self):
        for m in self.features.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        if self.classifier is not None:        
            for m in self.classifier.modules():
                if isinstance(m, nn.Conv2d):
                    nn.init.normal_(m.weight, std=0.01)
                    if m.bias is not None:
                        nn.init.zeros_(m.bias)


def squeezenet1_0(pretrained=False, **kwargs):
    model = SqueezeNet('1_0', **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['squeezenet1_0'], progress=True),
            strict=False)
    return model


def squeezenet1_1(pretrained=False, **kwargs):
    model = SqueezeNet('1_1', **kwargs)
    if pretrained:
        model.load_state_dict(
            load_url(model_urls['squeezenet1_1'], progress=True),
            strict=False)
    return model
