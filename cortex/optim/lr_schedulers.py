import math
from torch.optim.lr_scheduler import _LRScheduler
from bisect import bisect_right


__all__ = ['WarmupMultiStepLR', 'WarmupCosineLR']


def _get_warmup_factor(warmup_method,
                       warmup_steps,
                       warmup_factor,
                       last_step):
    if warmup_steps == 0 or last_step >= warmup_steps:
        return 1.0
    if warmup_method == 'constant':
        return warmup_factor
    elif warmup_method == 'linear':
        alpha = last_step / warmup_steps
        return warmup_factor * (1. - alpha) + alpha
    else:
        raise ValueError(
            'Unknown warmup method: {}'.format(warmup_method))


class WarmupMultiStepLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 milestones,
                 gamma=0.1,
                 warmup_factor=0.001,
                 warmup_steps=1000,
                 warmup_method='linear',
                 last_epoch=-1):
        if not list(milestones) == sorted(milestones):
            raise ValueError(
                'Milestones should be a list of increasing integers. '
                'Got {}'.format(milestones))
        self.milestones = milestones
        self.gamma = gamma
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        self.last_step = -1
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
        self.step_batch()
    
    def get_lr(self):
        warmup_factor = _get_warmup_factor(
            self.warmup_method,
            self.warmup_steps,
            self.warmup_factor,
            self.last_step)
        return [base_lr * warmup_factor * self.gamma ** \
            bisect_right(self.milestones, self.last_epoch)
            for base_lr in self.base_lrs]
    
    def step_batch(self, step=None):
        r"""Scheduler step applied after each batch is processed.
        """
        if self.last_step >= self.warmup_steps:
            return
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        for param_group, lr in zip(
            self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr


class WarmupCosineLR(_LRScheduler):

    def __init__(self,
                 optimizer,
                 max_epochs,
                 warmup_factor=0.001,
                 warmup_steps=1000,
                 warmup_method='linear',
                 last_epoch=-1):
        self.max_epochs = max_epochs
        self.warmup_factor = warmup_factor
        self.warmup_steps = warmup_steps
        self.warmup_method = warmup_method
        self.last_step = -1
        super(WarmupCosineLR, self).__init__(optimizer, last_epoch)
        self.step_batch()
    
    def get_lr(self):
        warmup_factor = _get_warmup_factor(
            self.warmup_method,
            self.warmup_steps,
            self.warmup_factor,
            self.last_step)
        return [base_lr * warmup_factor * 0.5 * \
            (1. + math.cos(math.pi * self.last_epoch / self.max_epochs))
            for base_lr in self.base_lrs]

    def step_batch(self, step=None):
        r"""Scheduler step applied after each batch is processed.
        """
        if self.last_step >= self.warmup_steps:
            return
        if step is None:
            step = self.last_step + 1
        self.last_step = step
        for param_group, lr in zip(
            self.optimizer.param_groups, self.get_lr()):
            param_group['lr'] = lr
