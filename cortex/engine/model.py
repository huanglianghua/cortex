import torch
import torch.nn as nn
from abc import abstractmethod
try:
    from apex import amp
except ImportError:
    amp = None


__all__ = ['Model']


class Model(nn.Module):

    @abstractmethod
    def build_dataloader(self, dataset, mode, state):
        raise NotImplementedError
    
    @abstractmethod
    def build_training(self, state):
        raise NotImplementedError

    @abstractmethod
    def train_step(self, batch, state):
        raise NotImplementedError
    
    @abstractmethod
    def val_step(self, batch, state):
        raise NotImplementedError
    
    @abstractmethod
    def test_step(self, batch, state):
        raise NotImplementedError
    
    def optimize_step(self, output, state):
        # check the format of output (from train_step)
        if not isinstance(output, dict):
            output = {'loss': output}
        assert isinstance(output, dict) and 'loss' in output
        assert isinstance(output['loss'], torch.Tensor) and \
            output['loss'].requires_grad
        
        # check mixed precision
        if state.cfg.mixed_precision and amp is None:
            raise ValueError(
                'Training with "mixed_precision=True" requires apex '
                'to be installed, but was not found')

        # parse state variables
        cfg = state.cfg
        optimizers = state.optimizers
        schedulers = state.schedulers
        logger = state.logger

        # reset gradients
        if isinstance(optimizers, list):
            for o in optimizers:
                o.zero_grad()
        else:
            optimizers.zero_grad()
        
        # run a step of backward pass, using the scaled loss
        loss = output['loss']
        if cfg.mixed_precision:
            with amp.scale_loss(loss, optimizers) as scaled_loss:
                scaled_loss.backward()
        else:
            loss.backward()
        
        # clip gradient if specified
        if cfg.max_grad_norm is not None:
            if cfg.mixed_precision:
                if isinstance(optimizers, list):
                    for o in optimizers:
                        torch.nn.utils.clip_grad_norm_(
                            amp.master_params(o),
                            cfg.max_grad_norm)
                else:
                    torch.nn.utils.clip_grad_norm_(
                        amp.master_params(optimizers),
                        cfg.max_grad_norm)
            else:
                params = []
                if isinstance(optimizers, list):
                    for o in optimizers:
                        for group in o.param_groups:
                            params += group['params']
                else:
                    for group in optimizers.param_groups:
                        params += group['params']
                torch.nn.utils.clip_grad_norm_(
                    filter(lambda p: p.requires_grad, params),
                    cfg.max_grad_norm)
        
        # optimization step
        if isinstance(optimizers, list):
            for o in optimizers:
                o.step()
        else:
            optimizers.step()
        
        # run a step (per-iter) of learning rate scheduler if available
        if schedulers is not None:
            cur_step = (state.epoch - 1) * state.epoch_size + state.step
            if isinstance(schedulers, list):
                for s in schedulers:
                    if hasattr(s, 'step_iter'):
                        s.step_iter(step=cur_step - 1)
            else:
                if hasattr(schedulers, 'step_iter'):
                    schedulers.step_iter(step=cur_step - 1)
        
        # check if it is the last step of this epoch
        if state.step == state.epoch_size and schedulers is not None:
            # run a step (per-epoch) of learning rate scheduler
            if isinstance(schedulers, list):
                for s in schedulers:
                    s.step(epoch=state.epoch - 1)
            else:
                schedulers.step(epoch=state.epoch - 1)
