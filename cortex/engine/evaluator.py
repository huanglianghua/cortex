import numbers
import logging
import torch
import torch.nn as nn
import os
import numpy as np
import random
import time
try:
    from apex import amp
except ImportError:
    amp = None

import cortex.data as data
import cortex.ops as ops
from .utils import Config, State
from .distributed import DataParallel, DistributedDataParallel


__all__ = ['run_evaluation']


def run_evaluation(model,
                   test_data,
                   test_metric=data.Average(),
                   distributed=False,
                   num_machines=1,
                   machine_rank=0,
                   gpus_per_machine=1,
                   master_addr='localhost',
                   master_port='8889',
                   cudnn_benchmark=True,
                   non_blocking=False,
                   mixed_precision=False,
                   opt_level='O2',
                   keep_batchnorm_fp32=None,
                   load_from=None,
                   echo_frequency=10,
                   deterministic=True,
                   manual_seed=None,
                   log_dir='runs/default_eval',
                   log_level=logging.DEBUG):
    # sanity check
    if mixed_precision and amp is None:
        raise ValueError(
            'Test with "mixed_precision=True" requires apex '
            'to be installed, but was not found')

    # GPU utilities
    cuda = torch.cuda.is_available() and gpus_per_machine != 0
    if not cuda:
        gpus_per_machine = 0
        cudnn_benchmark = False
        non_blocking = False
        mixed_precision = False
    elif gpus_per_machine is None:
        # use all available GPUs if not specified
        gpus_per_machine = torch.cuda.device_count()
    assert gpus_per_machine <= torch.cuda.device_count()
    distributed &= gpus_per_machine * num_machines > 1

    # initialize environment variables for distributed test
    if distributed:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port
    
    # store evaluation parameters
    cfg = Config()
    cfg.distributed = distributed
    cfg.num_machines = num_machines
    cfg.machine_rank = machine_rank
    cfg.gpus_per_machine = gpus_per_machine
    cfg.master_addr = master_addr
    cfg.master_port = master_port
    cfg.cudnn_benchmark = cudnn_benchmark
    cfg.non_blocking = non_blocking
    cfg.mixed_precision = mixed_precision
    cfg.opt_level = opt_level
    cfg.keep_batchnorm_fp32 = keep_batchnorm_fp32
    cfg.load_from = load_from
    cfg.echo_frequency = echo_frequency
    cfg.deterministic = deterministic
    cfg.manual_seed = manual_seed
    cfg.log_dir = log_dir
    cfg.log_level = log_level
    cfg.cuda = cuda

    # run evaluation
    args = (model, test_data, test_metric, cfg)
    if distributed:
        ops.launch(
            _single_process_test, args,
            gpus_per_machine, num_machines, machine_rank)
    else:
        _single_process_test(*args)


def _single_process_test(model, test_data, test_metric, cfg):
    # random (faster) or deterministic (slower) test
    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn_benchmark = cfg.cudnn_benchmark
        torch.backends.cudnn.deterministic = False
    
    # shared random seed across processes
    if cfg.manual_seed is None:
        cfg.manual_seed = ops.shared_random_seed()
    torch.manual_seed(cfg.manual_seed)
    np.random.seed(cfg.manual_seed)
    random.seed(cfg.manual_seed)

    # device and rank information
    rank = ops.get_rank()
    world_size = ops.get_world_size()
    device = torch.device('cuda' if cfg.cuda else 'cpu')
    distributed = cfg.cuda and cfg.distributed

    # initialize logger (should only work for rank 0)
    if rank == 0:
        logger = ops.Logger(path=cfg.log_dir, level=cfg.log_level)
    else:
        logger = ops.Logger(path=None)
    
    # log evaluation configurations
    logger.info(str(cfg))

    # initialize state variables
    state = State(
        rank=rank,
        world_size=world_size,
        cuda=cfg.cuda,
        device=device,
        distributed=distributed,
        logger=logger,
        cfg=cfg)
    
    # build test loader and update state variables
    logger.info('Building test dataloader...')
    test_loader = model.build_dataloader(
        test_data, mode='test', state=state)
    state.update(test_loader=test_loader)

    # load model from checkpoint if given
    model = model.to(device)
    if cfg.load_from is not None:
        logger.info('Loading from checkpoint: {}'.format(cfg.load_from))
        ops.load_checkpoint(
            cfg.load_from, model,
            map_location=device, logger=logger)
    
    # initialize mixed precision test
    if cfg.mixed_precision:
        logger.info('Initializing mixed precision test...')
        model = amp.initialize(
            model,
            opt_level=cfg.opt_level,
            keep_batchnorm_fp32=cfg.keep_batchnorm_fp32)
    
    # initialize parallel test
    if distributed:
        # distributed multi-gpu test
        model = DistributedDataParallel(model, device_ids=[device])
    elif cfg.cuda and cfg.gpus_per_machine > 1:
        # non-distributed multi-gpu test
        device_ids = list(range(cfg.gpus_per_machine))
        model = DataParallel(model, device_ids=device_ids)
    
    # initialize time measurement
    step_begin = None
    step_end = None
    time_metric = data.Average(default_name='time')

    # run test loop
    logger.info('Start test...')
    for step, batch in enumerate(test_loader, 1):
        model.eval()
        state.step = step

        # track running time
        if step_begin is None:
            step_begin = time.perf_counter()
            eta = None
        else:
            step_end = time.perf_counter()
            time_metric.update(step_end - step_begin)
            step_begin = step_end

            # ETA of evaluation
            avg_time = time_metric.compute()['time'].item()
            eta = avg_time * (len(test_loader) - step)

        # run a step of test (model.step)
        batch = ops.to_device(batch, device, cfg.non_blocking)
        with torch.no_grad():
            output = model.test_step(batch, state)
        
        if isinstance(model, DataParallel):
            outputs = output.split(1, dim=0)
        else:
            outputs = ops.gather(output, dst=0)
        
        # log metrics at frequency
        if rank == 0:
            # compute average metrics over processes/threads
            all_metrics = [test_metric.update(o) for o in outputs]
            if all([m is not None for m in all_metrics]):
                metrics = {
                    k: sum([m[k] for m in all_metrics])
                    for k in all_metrics[0]}
                metrics = {
                    k: v / len(outputs)
                    for k, v in metrics.items()}
            else:
                metrics = None
            
            # log test metrics
            if step == 1 or step % cfg.echo_frequency == 0 or \
                step == len(test_loader):
                # estimated time until arrival (test completed)
                if eta is None:
                    # the first step, no statistics
                    eta_str = 'N.A.'
                elif eta > 3600 * 24:
                    # more than a day
                    eta_str = time.strftime('%dd %Hh %Mm', time.gmtime(eta))
                else:
                    eta_str = time.strftime('%Hh %Mm %Ss', time.gmtime(eta))
                # message text
                msg = 'ETA: {} Test step: [{}/{}]'.format(
                    eta_str, step, len(test_loader))
                if metrics is not None:
                    for k, v in metrics.items():
                        msg += ' {}: {:.3f}'.format(k, float(v))
                logger.info(msg)
    
    # log test metrics over the complete test_loader
    if rank == 0:
        logger.info('Evaluating performance...')
        metrics = test_metric.compute()
        if metrics is not None:
            # log text message
            msg = 'Test completed!\n'
            for k, v in metrics.items():
                msg += ' {}: {:.3f}'.format(k, float(v))
            logger.info(msg)
        else:
            logger.warning('Got None type metrics')
    
    # destroy distributed process groups
    if distributed:
        ops.cleanup_dist()
