import importlib
import sys
import os
import logging
import numpy as np
import torch
import torch.nn as nn
import random
from datetime import datetime
from multiprocessing.pool import ThreadPool as Pool


def import_file(name, path, make_importable=False):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if make_importable:
        sys.modules[name] = module
    return module


def manual_seed(seed=None):
    if seed is None:
        # set a strong random seed if None
        seed = os.getpid() + \
            int(datetime.now().strftime('%S%f')) + \
            int.from_bytes(os.urandom(2), 'big')
        logger = logging.getLogger(__name__)
        logger.info('Using a generated random seed {}'.format(seed))
    np.random.seed(seed)
    torch.set_rng_state(torch.manual_seed(seed).get_state())
    random.seed(seed)


def to_device(batch, device, non_blocking=False):
    if isinstance(batch, (list, tuple)):
        return type(batch)([
            to_device(u, device, non_blocking)
            for u in batch])
    elif isinstance(batch, dict):
        return type(batch)([
            (k, to_device(v, device, non_blocking))
            for k, v in batch.items()])
    elif isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    return batch


def detach(data):
    if isinstance(data, (list, tuple)):
        return type(data)([detach(u) for u in data])
    elif isinstance(data, dict):
        return type(data)([
            (k, detach(v)) for k, v in data.items()])
    elif isinstance(data, torch.Tensor):
        return data.detach()
    return data


def map(func, args_list, num_workers=32, timeout=None):
    assert isinstance(args_list, list)
    if not isinstance(args_list[0], tuple):
        args_list = [(args, ) for args in args_list]
    with Pool(processes=num_workers) as pool:
        results = [pool.apply_async(func, args) for args in args_list]
        results = [res.get(timeout=timeout) for res in results]    
    return results


def broadcastable(a, b):
    return all([m == n or m == 1 or n == 1
                for m, n in zip(a.shape[::-1], b.shape[::-1])])


def init_weights(model):
    for m in model.modules():
        if isinstance(m, nn.modules.conv._ConvNd):
            nn.init.kaiming_normal_(
                m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, (
            nn.modules.batchnorm._BatchNorm, nn.GroupNorm)):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1.)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def distmat(x, y=None, sqrt=True, eps=1e-6):
    # calculate L2 distances
    x2 = x.pow(2).sum(1).view(-1, 1)
    if y is not None:
        y2 = y.pow(2).sum(1).view(1, -1)
        y_t = y.t()
    else:
        y2 = x2.view(1, -1)
        y_t = x.t()
    dist = x2 + y2 - 2.0 * torch.mm(x, y_t)

    # ensure diagonal is zero if x=y
    if y is None:
        dist = dist - torch.diag(dist.diag())
    dist = dist.clamp(0.0)

    # calculate the square root if sqrt=True
    if sqrt:
        mask = (dist == 0).float()
        dist = dist + mask * eps
        dist = dist.sqrt()
        dist = dist * (1.0 - mask)
    
    return dist


def logsumexp(x, mask=None, add_one=False, dim=-1):
    r"""log(sum(1 + mask * exp(x)))
    """
    if dim < 0:
        dim = x.ndim + dim
    assert x.ndim > dim
    assert x.ndim in [1, 2]

    if mask is not None:
        x = x.clone()
        x[mask.eq(0)] = -float('inf')
    
    if add_one:
        if x.ndim == 1:
            zeros = x.new_zeros(1)
        else:
            zeros = x.new_zeros(x.size(1 - dim)).unsqueeze(dim=dim)
        x = torch.cat([x, zeros], dim=dim)
    
    return torch.logsumexp(x, dim=dim)
