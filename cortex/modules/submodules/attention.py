import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['SEModule', 'CBAM']


class SEModule(nn.Module):

    def __init__(self, in_channels, reduction=4):
        super(SEModule, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels, in_channels // reduction, 1)
        self.expand = nn.Conv2d(
            in_channels // reduction, in_channels, 1)
    
    def forward(self, x):
        out = F.adaptive_avg_pool2d(x, 1)
        out = F.relu(self.reduce(out))
        out = torch.sigmoid(self.expand(out))
        return x * out


class _ChannelGate(nn.Module):

    def __init__(self, in_channels, reduction):
        super(_ChannelGate, self).__init__()
        self.reduce = nn.Conv2d(
            in_channels, in_channels // reduction, 1)
        self.expand = nn.Conv2d(
            in_channels // reduction, in_channels, 1)
    
    def forward(self, x):
        # squeeze-excitation on average pooling results
        x_avg = F.adaptive_avg_pool2d(x, 1)
        x_avg = self.expand(F.relu(self.reduce(x_avg)))
        # squeeze-excitation on max pooling results
        x_max = F.adaptive_max_pool2d(x, 1)
        x_max = self.expand(F.relu(self.reduce(x_max)))
        out = torch.sigmoid(x_avg + x_max)
        return x * out


class _SpatialGate(nn.Module):

    def __init__(self):
        super(_SpatialGate, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(2, 1, 7, stride=1, padding=3, bias=False),
            nn.BatchNorm2d(1, momentum=0.01, eps=1e-5, affine=True))
    
    def forward(self, x):
        out = torch.cat((
            x.max(dim=1)[0].unsqueeze(1),
            x.mean(dim=1).unsqueeze(1)), dim=1)
        out = self.conv(out)
        out = torch.sigmoid(out)
        return x * out


class CBAM(nn.Module):

    def __init__(self, in_channels, reduction=16):
        super(CBAM, self).__init__()
        self.channel_gate = _ChannelGate(in_channels, reduction)
        self.spatial_gate = _SpatialGate()
    
    def forward(self, x):
        x = self.channel_gate(x)
        x = self.spatial_gate(x)
        return x
