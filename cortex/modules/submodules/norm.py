import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ['FrozenBatchNorm2d', 'FilterResponseNorm1d',
           'FilterResponseNorm2d', 'FilterResponseNorm3d']


class FrozenBatchNorm2d(nn.Module):

    def __init__(self, num_features, eps=1e-5):
        super(FrozenBatchNorm2d, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.register_buffer('weight', torch.ones(num_features))
        self.register_buffer('bias', torch.zeros(num_features))
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer(
            'running_var', torch.ones(num_features) - eps)
    
    def forward(self, x):
        if x.requires_grad:
            # do not use F.batch_norm since it will use extra memory
            # for computing gradients
            scale = self.weight * (self.running_var + self.eps).rsqrt()
            bias = self.bias - self.running_mean * scale
            scale = scale.reshape(1, -1, 1, 1)
            bias = bias.reshape(1, -1, 1, 1)
            return x * scale + bias
        else:
            # F.batch_norm provides more optimization opportunities
            return F.batch_norm(
                x,
                self.running_mean,
                self.running_var,
                self.weight,
                self.bias,
                training=False,
                eps=self.eps)
    
    def _load_from_state_dict(
        self, state_dict, prefix, local_metadata, strict,
        missing_keys, unexpected_keys, error_msgs):
        version = local_metadata.get('version', None)

        if version is None or version < 2:
            if prefix + 'running_mean' not in state_dict:
                state_dict[prefix + 'running_mean'] = \
                    torch.zeros_like(self.running_mean)
            if prefix + 'running_var' not in state_dict:
                state_dict[prefix + 'running_var'] = \
                    torch.ones_like(self.running_var)

        super(FrozenBatchNorm2d, self)._load_from_state_dict(
            state_dict, prefix, local_metadata, strict,
            missing_keys, unexpected_keys, error_msgs)
    
    def __repr__(self):
        return 'FrozenBatchNorm2d(num_features={}, eps={})'.format(
            self.num_features, self.eps)


class FilterResponseNormNd(nn.Module):
    
    def __init__(self, ndim, num_features, eps=1e-6,
                 learnable_eps=False):
        assert ndim in [3, 4, 5], \
            'FilterResponseNorm only supports 3d, 4d or 5d inputs.'
        super(FilterResponseNormNd, self).__init__()
        shape = (1, num_features) + (1, ) * (ndim - 2)
        self.eps = nn.Parameter(torch.ones(*shape) * eps)
        if not learnable_eps:
            self.eps.requires_grad_(False)
        self.gamma = nn.Parameter(torch.Tensor(*shape))
        self.beta = nn.Parameter(torch.Tensor(*shape))
        self.tau = nn.Parameter(torch.Tensor(*shape))
        self.reset_parameters()
    
    def forward(self, x):
        avg_dims = tuple(range(2, x.dim()))
        nu2 = torch.pow(x, 2).mean(dim=avg_dims, keepdim=True)
        x = x * torch.rsqrt(nu2 + torch.abs(self.eps))
        return torch.max(self.gamma * x + self.beta, self.tau)

    def reset_parameters(self):
        nn.init.ones_(self.gamma)
        nn.init.zeros_(self.beta)
        nn.init.zeros_(self.tau)


class FilterResponseNorm1d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm1d, self).__init__(
            3, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm2d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm2d, self).__init__(
            4, num_features, eps=eps, learnable_eps=learnable_eps)


class FilterResponseNorm3d(FilterResponseNormNd):

    def __init__(self, num_features, eps=1e-6, learnable_eps=False):
        super(FilterResponseNorm3d, self).__init__(
            5, num_features, eps=eps, learnable_eps=learnable_eps)
