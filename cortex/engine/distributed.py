import torch
import torch.nn as nn
import torch.nn.parallel as parallel
import itertools
import threading
from torch.cuda._utils import _get_device_index
from torch._utils import ExceptionWrapper


__all__ = ['parallel_apply', 'DataParallel', 'DistributedDataParallel']


def _find_tensors(obj):
    if isinstance(obj, torch.Tensor):
        return [obj]
    if isinstance(obj, (list, tuple)):
        return itertools.chain(*map(_find_tensors, obj))
    if isinstance(obj, dict):
        return itertools.chain(*map(_find_tensors, obj.values()))
    return []


def get_a_var(obj):
    if isinstance(obj, torch.Tensor):
        return obj

    if isinstance(obj, list) or isinstance(obj, tuple):
        for result in map(get_a_var, obj):
            if isinstance(result, torch.Tensor):
                return result
    if isinstance(obj, dict):
        for result in map(get_a_var, obj.items()):
            if isinstance(result, torch.Tensor):
                return result
    return None


def parallel_apply(modules, inputs, kwargs_tup=None, devices=None,
                   mode=None):
    assert len(modules) == len(inputs)
    if kwargs_tup is not None:
        assert len(modules) == len(kwargs_tup)
    else:
        kwargs_tup = ({},) * len(modules)
    if devices is not None:
        assert len(modules) == len(devices)
    else:
        devices = [None] * len(modules)
    devices = list(map(lambda x: _get_device_index(x, True), devices))
    lock = threading.Lock()
    results = {}
    grad_enabled = torch.is_grad_enabled()

    def _worker(i, module, input, kwargs, device=None, mode=None):
        # parse forward function
        assert mode in [None, 'train', 'val', 'test']
        if mode is None:
            forward_fn = module.forward
        else:
            forward_fn = getattr(module, mode + '_step')

        torch.set_grad_enabled(grad_enabled)
        if device is None:
            device = get_a_var(input).get_device()
        try:
            with torch.cuda.device(device):
                if not isinstance(input, (list, tuple)):
                    input = (input,)
                output = forward_fn(*input, **kwargs)
            with lock:
                results[i] = output
        except Exception:
            with lock:
                results[i] = ExceptionWrapper(
                    where='in replica {} on device {}'.format(i, device))

    if len(modules) > 1:
        threads = [threading.Thread(
            target=_worker,
            args=(i, module, input, kwargs, device, mode))
            for i, (module, input, kwargs, device) in
            enumerate(zip(modules, inputs, kwargs_tup, devices))]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
    else:
        _worker(0, modules[0], inputs[0], kwargs_tup[0], devices[0],
                mode)

    outputs = []
    for i in range(len(inputs)):
        output = results[i]
        if isinstance(output, ExceptionWrapper):
            output.reraise()
        outputs.append(output)
    return outputs


class DataParallel(nn.DataParallel):

    def forward(self, *inputs, **kwargs):
        # parse forward function
        mode = kwargs.pop('mode', None)
        assert mode in [None, 'train', 'val', 'test']
        if mode is None:
            forward_fn = self.module.forward
        else:
            forward_fn = getattr(self.module, mode + '_step')

        if not self.device_ids:
            return forward_fn(*inputs, **kwargs)
        
        for t in itertools.chain(
            self.module.parameters(), self.module.buffers()):
            if t.device != self.src_device_obj:
                raise RuntimeError(
                    'module must have its parameters and buffers on '
                    'device {} (device_ids[0]) but found one of them on '
                    'device: {}'.format(self.src_device_obj, t.device))
        
        inputs, kwargs = self.scatter(inputs, kwargs, self.device_ids)
        if len(self.device_ids) == 1:
            return forward_fn(*inputs[0], **kwargs[0])
        replicas = self.replicate(
            self.module, self.device_ids[:len(inputs)])
        outputs = self.parallel_apply(
            replicas, inputs, kwargs, mode=mode)
        return self.gather(outputs, self.output_device)

    def train_step(self, *inputs, **kwargs):
        kwargs.update(mode='train')
        return self.forward(*inputs, **kwargs)
    
    def val_step(self, *inputs, **kwargs):
        kwargs.update(mode='val')
        return self.forward(*inputs, **kwargs)
    
    def test_step(self, *inputs, **kwargs):
        kwargs.update(mode='test')
        return self.forward(*inputs, **kwargs)
    
    def optimize_step(self, *outputs, **kwargs):
        return self.module.optimize_step(*outputs, **kwargs)

    def parallel_apply(self, replicas, inputs, kwargs, mode=None):
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[:len(replicas)],
            mode=mode)


class DistributedDataParallel(parallel.DistributedDataParallel):

    def forward(self, *inputs, **kwargs):
        # parse forward function
        mode = kwargs.pop('mode', None)
        assert mode in [None, 'train', 'val', 'test']
        if mode is None:
            forward_fn = self.module.forward
        else:
            forward_fn = getattr(self.module, mode + '_step')

        if self.require_forward_param_sync:
            self._sync_params()

        if self.device_ids:
            inputs, kwargs = self.scatter(
                inputs, kwargs, self.device_ids)
            if len(self.device_ids) == 1:
                output = forward_fn(*inputs[0], **kwargs[0])
            else:
                outputs = self.parallel_apply(
                    self._module_copies[:len(inputs)], inputs, kwargs,
                    mode=mode)
                output = self.gather(outputs, self.output_device)
        else:
            output = forward_fn(*inputs, **kwargs)
        
        if torch.is_grad_enabled() and self.require_backward_grad_sync:
            self.require_forward_param_sync = True
            if self.find_unused_parameters:
                self.reducer.prepare_for_backward(
                    list(_find_tensors(output)))
            else:
                self.reducer.prepare_for_backward([])
        else:
            self.require_forward_param_sync = False
        
        return output
    
    def train_step(self, *inputs, **kwargs):
        kwargs.update(mode='train')
        return self.forward(*inputs, **kwargs)
    
    def val_step(self, *inputs, **kwargs):
        kwargs.update(mode='val')
        return self.forward(*inputs, **kwargs)
    
    def test_step(self, *inputs, **kwargs):
        kwargs.update(mode='test')
        return self.forward(*inputs, **kwargs)
    
    def optimize_step(self, *outputs, **kwargs):
        return self.module.optimize_step(*outputs, **kwargs)

    def parallel_apply(self, replicas, inputs, kwargs, mode=None):
        return parallel_apply(
            replicas, inputs, kwargs, self.device_ids[:len(replicas)],
            mode=mode)
