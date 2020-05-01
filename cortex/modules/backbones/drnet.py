import torch
import torch.nn as nn


__all__ = ['DRNet']


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(
        in_channels, out_channels, 3, stride,
        padding=1, bias=False)


def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, 1, stride, bias=False)


class BasicBlock(nn.Module):

    def __init__(self, in_channels, channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(in_channels, channels, stride)
        self.bn1 = nn.BatchNorm2d(channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(channels, channels)
        self.bn2 = nn.BatchNorm2d(channels)
        self.downsample = None
        if stride != 1 or in_channels != channels:
            self.downsample = nn.Sequential(
                conv1x1(in_channels, channels, stride),
                nn.BatchNorm2d(channels))
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


class DRNet(nn.Module):
    r"""DRNet backbone from paper ``KPNet: towards minimal face detector''.
    """
    def __init__(self):
        super(DRNet, self).__init__()

        # build layer1~layer5
        self.layer1 = nn.Sequential(
            conv3x3(3, 64, stride=2),
            nn.BatchNorm2d(64))
        self.layer2 = nn.Sequential(
            BasicBlock(64, 64),
            BasicBlock(64, 64))
        self.layer3 = nn.Sequential(
            BasicBlock(64, 128, stride=2),
            BasicBlock(128, 128))
        self.layer4 = nn.Sequential(
            conv3x3(64 + 128, 64),
            nn.BatchNorm2d(64))
        self.layer5 = conv3x3(64 + 64, 128)

        # build pooling and upsampling layers
        self.maxpool = nn.MaxPool2d(2, stride=2)
        self.upsample = nn.Upsample(
            scale_factor=2, mode='bilinear', align_corners=True)
        
        # initialize weights
        self.init_weights()
    
    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(self.maxpool(x1))
        x3 = self.layer3(x2)
        x4 = self.layer4(torch.cat([x2, self.upsample(x3)], dim=1))
        x5 = self.layer5(torch.cat([x1, self.upsample(x4)], dim=1))
        return x5

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(
                    m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.)
                nn.init.zeros_(m.bias)
