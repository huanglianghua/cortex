import torch.nn as nn
from collections import OrderedDict
from torch.utils.model_zoo import load_url


__all__ = ['AlexNet', 'AlexNetV1', 'AlexNetV2', 'BigAlexNet', 'alexnet']


model_urls = {
    'alexnet': 'https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth'}


class _BatchNorm2d(nn.BatchNorm2d):

    def __init__(self, num_features, *args, **kwargs):
        super(_BatchNorm2d, self).__init__(
            num_features, *args,
            eps=1e-6, momentum=0.05, **kwargs)


class _AlexNet(nn.Module):

    def __init__(self, out_layers='conv5'):
        super(_AlexNet, self).__init__()
        self.out_layers = out_layers
    
    def forward(self, x, out_layers=None):
        if out_layers is None:
            out_layers = self.out_layers
        # check if returns single-layer output
        single_out = isinstance(out_layers, str)
        out = OrderedDict()
        
        # sequentially forward through conv1~conv5
        for layer in ['conv1', 'conv2', 'conv3', 'conv4', 'conv5']:
            x = getattr(self, layer)(x)
            if self._add_and_check(layer, x, out_layers, out):
                return out[out_layers] if single_out else out
    
    def _add_and_check(self, name, x, out_layers, out):
        if isinstance(out_layers, str):
            out_layers = [out_layers]
        if name in out_layers:
            out[name] = x
        return len(out_layers) == len(out)
    
    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.zeros_(m.bias)


class AlexNet(_AlexNet):

    def __init__(self, *args, **kwargs):
        super(AlexNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, 11, 4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(64, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(192, 384, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 256, 3, padding=1),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.init_weights()


class AlexNetV1(_AlexNet):

    def __init__(self, *args, **kwargs):
        super(AlexNetV1, self).__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 256, 3, 1, groups=2))
        self.init_weights()


class AlexNetV2(_AlexNet):

    def __init__(self, *args, **kwargs):
        super(AlexNetV2, self).__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 96, 11, 2),
            _BatchNorm2d(96),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(96, 256, 5, 1, groups=2),
            _BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 1))
        self.conv3 = nn.Sequential(
            nn.Conv2d(256, 384, 3, 1),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(384, 384, 3, 1, groups=2),
            _BatchNorm2d(384),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(384, 32, 3, 1, groups=2))
        self.init_weights()


class BigAlexNet(_AlexNet):

    def __init__(self, *args, **kwargs):
        super(BigAlexNet, self).__init__(*args, **kwargs)
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 192, 11, 2),
            _BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv2 = nn.Sequential(
            nn.Conv2d(192, 512, 5, 1),
            _BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, 2))
        self.conv3 = nn.Sequential(
            nn.Conv2d(512, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv4 = nn.Sequential(
            nn.Conv2d(768, 768, 3, 1),
            _BatchNorm2d(768),
            nn.ReLU(inplace=True))
        self.conv5 = nn.Sequential(
            nn.Conv2d(768, 512, 3, 1),
            _BatchNorm2d(512))
        self.init_weights()


def alexnet(pretrained=False, **kwargs):
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_url(model_urls['alexnet'], progress=True)
        model_state_dict = model.state_dict()
        for k1, k2 in zip(model_state_dict, state_dict):
            model_state_dict[k1] = state_dict[k2]
        model.load_state_dict(model_state_dict)
    return model
