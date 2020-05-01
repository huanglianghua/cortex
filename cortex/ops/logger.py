import logging
import sys
import os
import os.path as osp
import functools
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

from .distributed import get_rank


_LOGGING_METHODS = [
    'log', 'debug', 'info', 'warning', 'error', 'critical']

_SUMMARY_METHODS = [
    'add_hparams', 'add_scalar', 'add_scalars', 'add_histogram',
    'add_histogram_raw', 'add_image', 'add_images',
    'add_image_with_boxes', 'add_figure', 'add_video',
    'add_audio', 'add_text', 'add_onnx_graph', 'add_graph',
    'add_embedding', 'add_pr_curve', 'add_pr_curve_raw',
    'add_custom_scalars_multilinechart', 'add_mesh',
    'add_custom_scalars_marginchart', 'add_custom_scalars']

__all__ = ['Logger', 'set_default_logger', 'get_default_logger'] + \
    _LOGGING_METHODS + _SUMMARY_METHODS


def _rank0_decorator(func):
    # decroate func so that it only works at rank 0
    @functools.wraps(func)
    def wrapped(*args, **kwargs):
        if get_rank() == 0:
            return func(*args, **kwargs)
        return None
    return wrapped


class Logger(object):

    def __init__(self, name='cortex', path=None, level=logging.DEBUG):
        self._logger = logging.getLogger(name)
        self._logger.setLevel(level)
        self._logger.propagate = False
        self._writer = None

        # reset logging handlers
        for h in self._logger.handlers:
            self._logger.removeHandler(h)
        self._logger.handlers.clear()
        
        # console handler
        ch = logging.StreamHandler(stream=sys.stdout)
        ch.setLevel(level)
        ch.setFormatter(logging.Formatter(
            '[%(asctime)s] %(levelname)s: %(message)s',
            datefmt='%m.%d %H:%M:%S'))
        self._logger.addHandler(ch)

        # file handler and summary writer
        if path is not None:
            if path.endswith('.txt') or path.endswith('.log'):
                filename = path
            else:
                timestr = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
                filename = osp.join(path, timestr + '.log')
            dirname = osp.dirname(filename)
            if not osp.exists(dirname):
                os.makedirs(dirname)
            
            # file handler
            fh = logging.FileHandler(filename)
            fh.setLevel(level)
            fh.setFormatter(logging.Formatter(
                '[%(asctime)s] %(levelname)s: %(message)s',
                datefmt='%Y.%m.%d %H:%M:%S'))
            self._logger.addHandler(fh)
            
            # summary writer
            self._writer = SummaryWriter(log_dir=dirname)
        
        # assign logging methods (wrapped to only work for rank 0)
        for method in _LOGGING_METHODS:
            func = _rank0_decorator(getattr(self._logger, method))
            setattr(self, method, func)
        
        # assign summary methods (wrapped to only work for rank 0)
        for method in _SUMMARY_METHODS:
            func = getattr(self._writer, method,
                           self._not_implemented_error(method))
            func = _rank0_decorator(func)
            setattr(self, method, func)
    
    def _not_implemented_error(self, method):
        def _raise(*args, **kwargs):
            raise NotImplementedError(
                'To use function {}, please specify the path when '
                'initializing the Logger.'.format(method))
        return _raise
    
    def flush(self):
        for h in self._logger.handlers:
            h.flush()
        if self._writer is not None:
            self._writer.flush()
    
    def close(self):
        for h in self._logger.handlers:
            h.close()
        if self._writer is not None:
            self._writer.close()


# global default logger
root = Logger()

# define logging/summary methods based on the default logger
_func_template = 'def {}(*args, **kwargs):\n    root.{}(*args, **kwargs)'
for method in _LOGGING_METHODS + _SUMMARY_METHODS:
    exec(_func_template.format(method, method))


@functools.lru_cache(maxsize=None)
def set_default_logger(name='cortex', path=None, level=logging.DEBUG):
    global root
    root = Logger(name, path, level)
    return root


def get_default_logger():
    return root
