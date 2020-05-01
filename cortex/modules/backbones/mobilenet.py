import torch.nn as nn
from torch.utils.model_zoo import load_url


__all__ = ['MobileNetV2', 'mobilenet_v2']


model_urls = {
    'mobilenet_v2': 'https://download.pytorch.org/models/mobilenet_v2-b0353104.pth'}


def _make_divisible(val, divisor, min_ratio=0.9):
    new_val = max(divisor, int(val + divisor / 2)) // divisor * divisor
    # make sure that round down does not go down by more than 10%
    if new_val < min_ratio * val:
        new_val += divisor
    return new_val


class ConvBlock(nn.Sequential):

    def __init__(self, in_channels, out_channels, kernel_size=3,
                 stride=1, groups=1, activate=True):
        padding = (kernel_size - 1) // 2
        layers = [
            nn.Conv2d(in_channels, out_channels, kernel_size, stride,
                      padding, groups=groups, bias=False),
            nn.BatchNorm2d(out_channels)]
        if activate:
            layers.append(nn.ReLU6(inplace=True))
        super(ConvBlock, self).__init__(*layers)


class InvertedResidual(nn.Module):

    def __init__(self, in_channels, out_channels, stride, expansion):
        super(InvertedResidual, self).__init__()
        self.stride = stride
        assert stride in [1, 2]

        channels = int(round(in_channels * expansion))
        self.apply_residual = (self.stride == 1) and \
            (in_channels == out_channels)

        layers = []
        if expansion != 1:
            # pointwise convolution
            layers += [ConvBlock(in_channels, channels, 1)]
        layers += [
            # depthwise separable convolution
            ConvBlock(channels, channels, stride=stride, groups=channels),
            # pointwise convolution
            ConvBlock(channels, out_channels, 1, activate=False)]
        self.layers = nn.Sequential(*layers)
    
    def forward(self, x):
        if self.apply_residual:
            return x + self.layers(x)
        else:
            return self.layers(x)


class MobileNetV2(nn.Module):

    def __init__(self, num_classes=None, width_mult=1.,
                 inverted_residual_setting=None, round_nearest=8):
        super(MobileNetV2, self).__init__()
        in_channels = 32
        last_channels = 1280

        if inverted_residual_setting is None:
            inverted_residual_setting = [
                # t, c, n, s
                [1, 16, 1, 1],
                [6, 24, 2, 2],
                [6, 32, 3, 2],
                [6, 64, 4, 2],
                [6, 96, 3, 1],
                [6, 160, 3, 2],
                [6, 320, 1, 1]]
        
        if len(inverted_residual_setting) == 0 or \
            len(inverted_residual_setting[0]) != 4:
            raise ValueError(
                'inverted_residual_setting should be non-empty '
                'or a 4-element list, got {}'.format(
                    inverted_residual_setting))
        
        # build the first layer
        in_channels = _make_divisible(
            in_channels * width_mult, round_nearest)
        self.last_channels = _make_divisible(
            last_channels * max(1., width_mult), round_nearest)
        features = [ConvBlock(3, in_channels, stride=2)]
        # build inverted residual blocks
        for t, c, n, s in inverted_residual_setting:
            out_channels = _make_divisible(c * width_mult, round_nearest)
            for i in range(n):
                stride = s if i == 0 else 1
                features += [InvertedResidual(
                    in_channels, out_channels, stride, expansion=t)]
                in_channels = out_channels
        # build the last several layers
        features += [ConvBlock(in_channels, self.last_channels, 1)]
        self.features = nn.Sequential(*features)

        # build classifier if num_classes is specified
        if num_classes is not None:
            self.classifier = nn.Sequential(
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Dropout(p=0.2),
                nn.Linear(self.last_channels, num_classes))
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
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)


def mobilenet_v2(pretrained=False, **kwargs):
    model = MobileNetV2(**kwargs)
    if pretrained:
        state_dict = load_url(model_urls['mobilenet_v2'], progress=True)
        model_state_dict = model.state_dict()
        for k1, k2 in zip(model_state_dict, state_dict):
            model_state_dict[k1] = state_dict[k2]
        model.load_state_dict(model_state_dict)
    return model
