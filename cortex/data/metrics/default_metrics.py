import numbers
import torch

import cortex.ops as ops


__all__ = ['Metric', 'Average']


class Metric(object):

    def reset(self):
        raise NotImplementedError
    
    def update(self, output):
        raise NotImplementedError
    
    def compute(self):
        raise NotImplementedError


class Average(Metric):

    def __init__(self, default_name='loss', detach=True):
        self.default_name = default_name
        self.detach = detach
        self.reset()
    
    def reset(self):
        self.accumulator = None
        self.num_samples = 0.
    
    def update(self, output):
        output = self._check_output(output)

        # always detach the variable when accumulating
        if self.accumulator is None:
            self.accumulator = ops.detach(output)
        elif isinstance(output, dict):
            for k, v in output.items():
                assert k in self.accumulator
                self.accumulator[k] += ops.detach(v)
        else:
            self.accumulator += ops.detach(output)
        self.num_samples += 1.

        # return current metrics
        if isinstance(output, dict):
            metrics = type(output)()
            for k, v in output.items():
                if self.detach:
                    v = ops.detach(v)
                metrics[k] = v
        else:
            if self.detach:
                output = ops.detach(output)
            metrics = {self.default_name: output}
        
        return metrics
    
    def compute(self):
        if isinstance(self.accumulator, dict):
            metrics = type(self.accumulator)()
            for k, v in self.accumulator.items():
                metrics[k] = v / float(self.num_samples)
            return metrics
        else:
            metrics = self.accumulator / float(self.num_samples)
            return {self.default_name: metrics}
    
    def _check_output(self, output):
        def _check_number(x):
            # ensure the output to be a 1-element tensor
            if isinstance(x, numbers.Number):
                x = torch.Tensor([x])[0]
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    'Output should be a (or a dict of) number or '
                    '1-element torch.Tensor. Got {}'.format(type(x)))
            elif x.numel() > 1:
                raise ValueError(
                    'Only 1-element torch.Tensor is supported. '
                    'Got {}'.format(x.size()))
            return x
        
        if isinstance(output, dict):
            return type(output)([
                (k, _check_number(v)) for k, v in output.items()])
        else:
            return _check_number(output)
