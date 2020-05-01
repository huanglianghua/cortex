import os
import os.path as osp
import time
import torch
import torch.nn as nn
import logging
from collections import OrderedDict


def save_checkpoint(filename,
                    model,
                    criterions=None,
                    optimizers=None,
                    meta_dict=None):
    dirname = osp.dirname(filename) or '.'
    if not osp.exists(dirname):
        os.makedirs(dirname)
    checkpoint = {}

    # checkpoint of meta
    if meta_dict is None:
        meta_dict = {}
    elif not isinstance(meta_dict, dict):
        raise TypeError(
            'meta_dict must be a dict or None, '
            'but got {}'.format(type(meta_dict)))
    meta_dict.update(time=time.asctime())
    checkpoint['meta'] = meta_dict

    # checkpoint of model
    if isinstance(model, (nn.DataParallel, \
        nn.parallel.DistributedDataParallel)):
        model = model.module
    state_dict = OrderedDict()
    for k, v in model.state_dict().items():
        state_dict[k] = v.cpu()
    checkpoint['state_dict'] = state_dict

    # checkpoint of criterions
    if isinstance(criterions, nn.Module):
        state_dict = OrderedDict()
        for k, v in criterions.state_dict().items():
            state_dict[k] = v.cpu()
        checkpoint['criterions'] = state_dict
    elif isinstance(criterions, list):
        state_dict = [OrderedDict()] * len(criterions)
        for i, m in enumerate(criterions):
            assert isinstance(m, nn.Module)
            for k, v in m.state_dict().items():
                state_dict[i][k] = v.cpu()
        checkpoint['criterions'] = state_dict
    elif criterions is not None:
        raise TypeError(
            'criterions must be a Module or a list of Modules, '
            'but got {}'.format(type(criterions)))
    
    # checkpoint of optimizers
    if isinstance(optimizers, torch.optim.Optimizer):
        checkpoint['optimizers'] = optimizers.state_dict()
    elif isinstance(optimizers, list):
        checkpoint['optimizers'] = [o.state_dict() for o in optimizers]
    elif optimizers is not None:
        raise TypeError(
            'optimizers must be an Optimizer or a list of Optimizers, '
            'but got {}'.format(type(optimizers)))
    
    # save to checkpoint file
    torch.save(checkpoint, filename)


def load_checkpoint(filename,
                    model,
                    criterions=None,
                    optimizers=None,
                    map_location=None,
                    strict=False,
                    logger=None):
    # load checkpoint file
    if map_location is None:
        map_location = next(model.parameters()).device
    checkpoint = torch.load(filename, map_location=map_location)
    if isinstance(checkpoint, OrderedDict):
        state_dict = checkpoint
    elif isinstance(checkpoint, dict) and 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    else:
        raise RuntimeError(
            'No state_dict found in checkpoint file {}' \
            .format(filename))
    
    # strip 'module.' prefix
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = type(state_dict)([
            (k[7:], v) for k, v in state_dict.items()])
    
    # load checkpoint of model
    if isinstance(model, (nn.DataParallel, \
        nn.parallel.DistributedDataParallel)):
        _load_state_dict(model.module, state_dict, strict, logger)
    else:
        _load_state_dict(model, state_dict, strict, logger)
    
    # load checkpoint of criterions
    if criterions is not None and 'criterions' in checkpoint:
        state_dict = checkpoint['criterions']
        if isinstance(criterions, nn.Module):
            _load_state_dict(criterions, state_dict, True, logger)
        elif isinstance(criterions, list):
            assert isinstance(state_dict, list)
            for i, m in enumerate(criterions):
                assert isinstance(m, nn.Module)
                _load_state_dict(
                    criterions[i], state_dict[i], True, logger)
        else:
            raise TypeError(
                'criterions must be a Module or a list of Modules, '
                'but got {}'.format(type(criterions)))
    
    # load checkpoint of optimizers
    if optimizers is not None and 'optimizers' in checkpoint:
        state_dict = checkpoint['optimizers']
        if isinstance(optimizers, torch.optim.Optimizer):
            _load_state_dict(optimizers, state_dict, True, logger)
        elif isinstance(optimizers, list):
            assert isinstance(state_dict, list)
            for i, m in enumerate(optimizers):
                assert isinstance(m, torch.optim.Optimizer)
                _load_state_dict(
                    optimizers[i], state_dict[i], True, logger)
        else:
            raise TypeError(
                'optimizers must be an Optimizer or a list of '
                'Optimizers, but got {}'.format(type(optimizers)))
    
    return checkpoint


def _load_state_dict(model, state_dict, strict=False, logger=None):
    own = model.state_dict()
    missing_keys = [k for k in own if not k in state_dict]
    unexpected_keys = [k for k in state_dict if not k in own]
    shape_mismatch_pairs = []
    
    # copy state_dict items
    for k, v in state_dict.items():
        if k in unexpected_keys:
            continue
        if isinstance(v, nn.Parameter):
            v = v.data
        if isinstance(v, torch.Tensor) and v.size() != own[k].size():
            shape_mismatch_pairs.append(
                [k, own[k].size(), v.size()])
            continue
        own[k] = v
    model.load_state_dict(own)
    
    # error messages
    err_msg = []
    if unexpected_keys:
        err_msg.append(
            'unexpected keys in source state_dict: {}\n'.format(
                ', '.join(unexpected_keys)))
    if missing_keys:
        err_msg.append(
            'missing keys in source state_dict: {}\n'.format(
                ', '.join(missing_keys)))
    if shape_mismatch_pairs:
        err_msg.extend([
            'these keys have mismatched shape:\n',
            'key | expected shape | loaded shape'])
        err_msg.extend([
            '{} | {} | {}'.format(*item)
            for item in shape_mismatch_pairs])
    
    # log error messages
    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state_dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        else:
            if logger is None:
                logger = logging.getLogger(__name__)
            logger.warning(err_msg)
