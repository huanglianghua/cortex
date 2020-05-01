import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict


__all__ = ['FPN']


class FPN(nn.Module):

    def __init__(self, in_channels, out_channels, num_outputs=None,
                 interp='nearest', downsample='maxpool'):
        super(FPN, self).__init__()
        # in_channels is a list of depths of multi-layer input features
        assert isinstance(in_channels, list)
        assert interp in ['nearest', 'linear', 'bilinear', 'bicubic']
        assert downsample in ['maxpool', 'conv']
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.interp = interp
        self.downsample = downsample
        self.num_inputs = len(in_channels)
        if num_outputs is None:
            num_outputs = self.num_inputs
        self.num_outputs = num_outputs

        # build lateral (1x1) and output (3x3) convolutional layers
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_channels[i], out_channels, 1)
            for i in range(self.num_inputs)])
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(self.num_inputs)])
        
        # extra output layers (only for stride-conv downsampling)
        num_extra = self.num_outputs - len(self.output_convs)
        if downsample == 'conv' and num_extra >= 1:
            for i in range(num_extra):
                self.output_convs.append(nn.Conv2d(
                    out_channels, out_channels, 3, 2, padding=1))
        
        # initialize weights
        self.init_weights()
    
    def forward(self, inputs):
        assert isinstance(inputs, (list, OrderedDict))
        if isinstance(inputs, OrderedDict):
            inputs = list(inputs.values())
        assert len(inputs) == self.num_inputs

        # lateral convolutions (1x1)
        laterals = [conv(input) for conv, input in zip(
            self.lateral_convs, inputs)]
        
        # upsample and add
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i - 1] += F.interpolate(
                laterals[i],
                size=laterals[i - 1].shape[2:],
                mode=self.interp)
        
        # fpn outputs (3x3)
        outs = [conv(lateral) for conv, lateral in zip(
            self.output_convs, laterals)]
        
        # generate extra outputs by downsampling
        # (by pooling or strided-conv)
        if self.num_outputs > len(outs):
            num_used = len(outs)
            num_extra = self.num_outputs - num_used
            if self.downsample == 'maxpool':
                for i in range(num_extra):
                    outs += [F.max_pool2d(outs[-1], 1, stride=2)]
            elif self.downsample == 'conv':
                outs += [self.output_convs[num_used](outs[-1])]
                for i in range(num_used + 1, self.num_outputs):
                    outs += [self.output_convs[i](F.relu(outs[-1]))]
        
        return tuple(outs)

    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform_(m.weight, gain=1.)
                nn.init.zeros_(m.bias)
