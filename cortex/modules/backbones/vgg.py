import torch
import torch.nn as nn
import numpy as np
from torch.utils.model_zoo import load_url


__all__ = ['VGG', 'VGGAtrous', 'vgg11', 'vgg11_bn', 'vgg13', 'vgg13_bn',
           'vgg16', 'vgg16_bn', 'vgg19', 'vgg19_bn', 'ssd300_vgg',
           'ssd512_vgg']


cfgs = {
    11: [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    13: [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    16: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    19: [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M']}

extra_cfgs = {
    300: [256, 'S', 512, 128, 'S', 256, 128, 256, 128, 256],
    512: [256, 'S', 512, 128, 'S', 256, 128, 'S', 256, 128, 'S', 256, 128, 'K4', 256]}

model_urls = {
    'vgg11': 'https://download.pytorch.org/models/vgg11-bbd30ac9.pth',
    'vgg13': 'https://download.pytorch.org/models/vgg13-c768596a.pth',
    'vgg16': 'https://download.pytorch.org/models/vgg16-397923af.pth',
    'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth',
    'vgg11_bn': 'https://download.pytorch.org/models/vgg11_bn-6002323d.pth',
    'vgg13_bn': 'https://download.pytorch.org/models/vgg13_bn-abd245e5.pth',
    'vgg16_bn': 'https://download.pytorch.org/models/vgg16_bn-6c64b313.pth',
    'vgg19_bn': 'https://download.pytorch.org/models/vgg19_bn-c79401a0.pth'}


def make_layers(cfg, batch_norm=False, ceil_mode=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(2, 2, ceil_mode=ceil_mode)]
        else:
            if batch_norm:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1),
                    nn.BatchNorm2d(v),
                    nn.ReLU(inplace=True)]
            else:
                layers += [
                    nn.Conv2d(in_channels, v, 3, padding=1),
                    nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


def make_extra_layers(cfg, in_channels=1024):
    layers = []
    for i, v in enumerate(cfg):
        if isinstance(v, str):
            continue
        else:
            if i > 0 and cfg[i - 1] == 'S':
                layers += [
                    nn.Conv2d(in_channels, v, 3, stride=2, padding=1),
                    nn.ReLU(inplace=True)]
            elif i > 0 and str(cfg[i - 1]).startswith('K'):
                kernel_size = int(cfg[i - 1][1])
                padding = (kernel_size - 1) // 2
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size, stride=1,
                              padding=padding)]
            else:
                kernel_size = 1 if in_channels > v else 3
                layers += [
                    nn.Conv2d(in_channels, v, kernel_size),
                    nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class _L2Norm(nn.Module):

    def __init__(self, in_channels, gamma=20., eps=1e-10):
        super(_L2Norm, self).__init__()
        self.in_channels = in_channels
        self.gamma = gamma
        self.eps = eps
        # learnable channel weights
        self.channel_weights = nn.Parameter(
            torch.FloatTensor(self.in_channels))
        nn.init.constant_(self.channel_weights, gamma)
    
    def forward(self, x):
        x = x.div(x.norm(dim=1, keepdim=True) + self.eps)
        x = self.channel_weights.view(1, -1, 1, 1) * x
        return x


class VGG(nn.Module):

    def __init__(self, depth, batch_norm=False, num_classes=None):
        super(VGG, self).__init__()
        self.features = make_layers(cfgs[depth], batch_norm)

        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((7, 7)),
                nn.Flatten(),
                nn.Linear(512 * 7 * 7, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, num_classes))
        else:
            self.classifier = None

        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        x = self.features(x)
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
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


class VGGAtrous(nn.Module):

    def __init__(self, depth, input_size, batch_norm=False):
        assert input_size in [300, 512]
        super(VGGAtrous, self).__init__()
        self.depth = depth
        self.input_size = input_size
        self.batch_norm = batch_norm
        self.features = make_layers(
            cfgs[depth][:-1],  # drop the last max-pooling
            batch_norm, ceil_mode=True)
        self.fc = nn.Sequential(
            nn.MaxPool2d(3, 1, padding=1),
            nn.Conv2d(512, 1024, 3, padding=6, dilation=6),
            nn.ReLU(inplace=True),
            nn.Conv2d(1024, 1024, 1),
            nn.ReLU(inplace=True))
        self.extra = make_extra_layers(
            extra_cfgs[input_size], in_channels=1024)
        self.l2_norm = _L2Norm(512, gamma=20)
        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        out = []
        
        # find index of the first output layer
        maxpools = [isinstance(u, nn.MaxPool2d) for u in self.features]
        last_pool = np.where(maxpools)[0][-1]

        # output layer 1
        x = self.features[:last_pool](x)
        out.append(self.l2_norm(x))

        # output layer 2
        x = self.features[last_pool:](x)
        x = self.fc(x)
        out.append(x)

        # output layer 3~6 (for ssd300) or 3~7 (for ssd512)
        num_layers = (len(self.extra) + 3) // 4
        for l in range(num_layers):
            x = self.extra[l * 4:(l + 1) * 4](x)
            out.append(x)
        
        return out
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)


def _load_pretrained(name, model):
    assert name in model_urls
    key_dict = {
        'classifier.0.': 'classifier.2.',
        'classifier.3.': 'classifier.5.',
        'classifier.6.': 'classifier.8.'}
    src_state = load_url(model_urls[name], progress=True)
    dst_state = type(src_state)()
    for k, v in src_state.items():
        for k1, k2 in key_dict.items():
            k = k.replace(k1, k2)
        dst_state[k] = v
    return model.load_state_dict(dst_state, strict=False)


def vgg11(pretrained=False, **kwargs):
    model = VGG(depth=11, batch_norm=False, **kwargs)
    if pretrained:
        _load_pretrained('vgg11', model)
    return model


def vgg11_bn(pretrained=False, **kwargs):
    model = VGG(depth=11, batch_norm=True, **kwargs)
    if pretrained:
        _load_pretrained('vgg11_bn', model)
    return model


def vgg13(pretrained=False, **kwargs):
    model = VGG(depth=13, batch_norm=False, **kwargs)
    if pretrained:
        _load_pretrained('vgg13', model)
    return model


def vgg13_bn(pretrained=False, **kwargs):
    model = VGG(depth=13, batch_norm=True, **kwargs)
    if pretrained:
        _load_pretrained('vgg13_bn', model)
    return model


def vgg16(pretrained=False, **kwargs):
    model = VGG(depth=16, batch_norm=False, **kwargs)
    if pretrained:
        _load_pretrained('vgg16', model)
    return model


def vgg16_bn(pretrained=False, **kwargs):
    model = VGG(depth=16, batch_norm=True, **kwargs)
    if pretrained:
        _load_pretrained('vgg16_bn', model)
    return model


def vgg19(pretrained=False, **kwargs):
    model = VGG(depth=19, batch_norm=False, **kwargs)
    if pretrained:
        _load_pretrained('vgg19', model)
    return model


def vgg19_bn(pretrained=False, **kwargs):
    model = VGG(depth=19, batch_norm=True, **kwargs)
    if pretrained:
        _load_pretrained('vgg19_bn', model)
    return model


def ssd300_vgg(pretrained=False):
    model = VGGAtrous(depth=16, input_size=300, batch_norm=False)
    if pretrained:
        _load_pretrained('vgg16', model)
    return model


def ssd512_vgg(pretrained=False):
    model = VGGAtrous(depth=16, input_size=512, batch_norm=False)
    if pretrained:
        _load_pretrained('vgg16', model)
    return model
