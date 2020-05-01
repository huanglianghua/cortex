import numbers
import logging
import torch
import torch.nn as nn
import os
import os.path as osp
import numpy as np
import random
import time
from collections import OrderedDict
try:
    from apex import amp
except ImportError:
    amp = None

import cortex.data as data
import cortex.ops as ops
from .utils import Config, State
from .distributed import DataParallel, DistributedDataParallel


__all__ = ['run_train']


def run_train(model,
              train_data,
              train_metric=data.Average(),
              val_data=None,
              val_metric=data.Average(),
              best_indicator=None,
              num_epochs=1,
              start_epoch=1,
              distributed=False,
              num_machines=1,
              machine_rank=0,
              gpus_per_machine=1,
              master_addr='localhost',
              master_port='8888',
              cudnn_benchmark=False,
              non_blocking=False,
              mixed_precision=False,
              opt_level='O2',
              loss_scale=None,
              keep_batchnorm_fp32=None,
              max_grad_norm=None,
              load_from=None,
              resume_from=None,
              echo_frequency=10,
              save_frequency=None,
              val_frequency=None,
              deterministic=False,
              manual_seed=None,
              log_dir='runs/default_train',
              log_level=logging.DEBUG):
    # sanity check
    if save_frequency is None:
        save_frequency = max(1, num_epochs // 10)
    if val_frequency is None:
        val_frequency = max(1, num_epochs // 10)
    if mixed_precision and amp is None:
        raise ValueError(
            'Training with "mixed_precision=True" requires apex '
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

    # initialize environment variables for distributed training
    if distributed:
        os.environ['MASTER_ADDR'] = master_addr
        os.environ['MASTER_PORT'] = master_port

    # store training parameters
    cfg = Config()
    cfg.best_indicator = best_indicator
    cfg.num_epochs = num_epochs
    cfg.start_epoch = start_epoch
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
    cfg.loss_scale = loss_scale
    cfg.keep_batchnorm_fp32 = keep_batchnorm_fp32
    cfg.max_grad_norm = max_grad_norm
    cfg.load_from = load_from
    cfg.resume_from = resume_from
    cfg.echo_frequency = echo_frequency
    cfg.save_frequency = save_frequency
    cfg.val_frequency = val_frequency
    cfg.deterministic = deterministic
    cfg.manual_seed = manual_seed
    cfg.log_dir = log_dir
    cfg.log_level = log_level
    cfg.cuda = cuda

    # run training
    args = (model, train_data, train_metric, val_data, val_metric, cfg)
    if distributed:
        ops.launch(
            _single_process_train, args,
            gpus_per_machine, num_machines, machine_rank)
    else:
        _single_process_train(*args)


def _single_process_train(
    model, train_data, train_metric, val_data, val_metric, cfg):
    # random (faster) or deterministic (slower) training
    if cfg.deterministic:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
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

    # log training configurations
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

    # build training loader
    logger.info('Building dataloaders...')
    train_loader = model.build_dataloader(
        train_data, mode='train', state=state)
    train_iterator = iter(train_loader)

    # build validation loader if available
    val_loader = None
    if val_data is not None:
        val_loader = model.build_dataloader(
            val_data, mode='val', state=state)

    # update state variables
    state.update(
        train_loader=train_loader,
        val_loader=val_loader)

    # build criterions, optimizers and lr schedulers
    logger.info('Building criterions, optimizers and lr schedulers...')
    criterions, optimizers, schedulers = model.build_training(state)

    # move to device
    model = model.to(device)
    if isinstance(criterions, nn.Module):
        criterions = criterions.to(device)
    elif isinstance(criterions, list):
        criterions = [u.to(device) for u in criterions]

    # load or resume from checkpoint
    if cfg.resume_from is not None:
        logger.info('Resuming from checkpoint: {}'.format(
            cfg.resume_from))
        checkpoint = ops.load_checkpoint(
            cfg.resume_from, model, criterions, optimizers,
            map_location=device, logger=logger)
        if isinstance(checkpoint, dict) and \
            not isinstance(checkpoint, OrderedDict) and \
            'epoch' in checkpoint.get('meta', {}):
            logger.info('Resuming epoch: {}'.format(
                checkpoint['meta']['epoch']))
            cfg.start_epoch = checkpoint['meta']['epoch'] + 1
        # resume lr scheduler if available
        if schedulers is not None:
            if isinstance(schedulers, list):
                for s in schedulers:
                    s.step(epoch=cfg.start_epoch - 1)
            else:
                schedulers.step(epoch=cfg.start_epoch - 1)
    elif cfg.load_from is not None:
        logger.info('Loading from checkpoint: {}'.format(
            cfg.load_from))
        ops.load_checkpoint(
            cfg.load_from, model,
            map_location=device, logger=logger)

    # initialize mixed precision training
    if cfg.mixed_precision:
        logger.info('Initializing mixed precision training...')
        model, optimizers = amp.initialize(
            model,
            optimizers,
            opt_level=cfg.opt_level,
            keep_batchnorm_fp32=cfg.keep_batchnorm_fp32,
            loss_scale=cfg.loss_scale)

    # update state variables
    state.update(
        criterions=criterions,
        optimizers=optimizers,
        schedulers=schedulers)

    # initialize parallel training
    if distributed:
        # distributed multi-gpu training
        model = DistributedDataParallel(model, device_ids=[device])
    elif cfg.cuda and cfg.gpus_per_machine > 1:
        # non-distributed multi-gpu training
        device_ids = list(range(cfg.gpus_per_machine))
        model = DataParallel(model, device_ids=device_ids)

    # calculate epoch_size (training steps per epoch)
    try:
        epoch_size = len(train_loader)
    except:
        # e.g., infinite samplers has no __len__
        assert train_loader._auto_collation
        if hasattr(train_loader.sampler, '__len__'):
            dataset_size = len(train_loader.sampler)
        elif hasattr(train_loader.sampler, 'length'):
            dataset_size = train_loader.sampler.length
        else:
            logger.warning(
                'Using "len(dataset) // world_size" to determine '
                'sampler size, which could be inaccurate')
            dataset_size = len(train_loader.dataset) // state.world_size
        # calculate epoch_size according to dataset (sampler) size
        batch_size = train_loader.batch_size
        if train_loader.drop_last:
            epoch_size = dataset_size // batch_size
        else:
            epoch_size = (dataset_size + batch_size - 1) // batch_size
    state.epoch_size = epoch_size

    # initialize time measurement
    step_begin = None
    step_end = None
    time_metric = data.Average(default_name='time')
    num_steps = cfg.num_epochs * epoch_size

    # initialize the highest value of best_indicator
    if cfg.best_indicator is not None:
        state.best_indicator = cfg.best_indicator
        state.best_value = -float('inf')
    else:
        state.best_indicator = None
        state.best_value = None

    # run training loop
    logger.info('Start training...')
    state.epoch = cfg.start_epoch
    state.step = 1
    while state.epoch <= cfg.num_epochs:
        # set to training mode
        model.train()

        # track running time
        if step_begin is None:
            step_begin = time.perf_counter()
            eta = None
        else:
            step_end = time.perf_counter()
            elapsed_time = time_metric.update(step_end - step_begin)
            elapsed_time = elapsed_time['time'].item()
            step_begin = step_end

            # ETA of training
            cur_step = (state.epoch - 1) * epoch_size + state.step
            eta = elapsed_time * (num_steps - cur_step + 1)

        # sample batch data and move to device
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_loader)
            batch = next(train_iterator)
        batch = ops.to_device(batch, device, cfg.non_blocking)

        # run a step of forward pass (model.train_step)
        with torch.enable_grad():
            output = model.train_step(batch, state)
        assert isinstance(output, (dict, torch.Tensor))

        # for DataParallel model, average over all outputs
        if isinstance(model, DataParallel):
            if isinstance(output, torch.Tensor):
                output = output.mean(dim=0)
            elif isinstance(output, dict):
                for k, v in output.items():
                    if isinstance(v, torch.Tensor):
                        output[k] = v.mean(dim=0)

        # ensure "loss" in output keys
        if not isinstance(output, dict):
            output = {'loss': output}
        assert 'loss' in output and \
            isinstance(output['loss'], torch.Tensor)

        # run a step of optimization
        model.optimize_step(output, state)

        # log metrics at frequency
        metrics = train_metric.update(output)
        metrics = ops.reduce_dict(metrics, reduction='mean')
        if rank == 0:
            # log scalars
            for k, v in metrics.items():
                k = 'train/' + k
                global_step = state.epoch_size * (state.epoch - 1) + state.step
                logger.add_scalar(k, v, global_step=global_step)

            # log text message
            if state.step == 1 or \
                state.step % cfg.echo_frequency == 0 or \
                state.step == epoch_size:
                # estimated time until arrival (training completed)
                if eta is None:
                    # the first step, no statistics
                    eta_str = 'N.A.'
                elif eta > 3600 * 24:
                    # more than a day
                    d, s = divmod(eta, 3600 * 24)
                    h, s = divmod(s, 3600)
                    m, s = divmod(s, 60)
                    eta_str = '%02dd %02dh %02dm' % (d, h, m)
                else:
                    # less than a day
                    h, s = divmod(eta, 3600)
                    m, s = divmod(s, 60)
                    eta_str = '%02dh %02dm %02ds' % (h, m, s)
                # message text
                msg = 'ETA: {} Epoch: [{}/{}] Step: [{}/{}] ' \
                      'Loss: {:.3f}'.format(
                    eta_str, state.epoch, cfg.num_epochs, state.step,
                    epoch_size, float(metrics.pop('loss')))
                # append other available metrics
                for k, v in metrics.items():
                    msg += ' {}: {:.3f}'.format(k, float(v))
                logger.info(msg)

        # check if epoch ends
        if state.step == epoch_size:
            # save the latest checkpoint after every epoch ends
            if rank == 0:
                filename = osp.join(cfg.log_dir, 'latest.pth')
                meta = {'epoch': state.epoch}
                ops.save_checkpoint(
                    filename, model, criterions, optimizers, meta)
                logger.info('Checkpoint saved at {}'.format(filename))

            # save checkpoints at frequency
            if rank == 0 and (
                state.epoch % cfg.save_frequency == 0 or \
                state.epoch == cfg.num_epochs):
                filename = osp.join(
                    cfg.log_dir, 'epoch_{}.pth'.format(state.epoch))
                meta = {'epoch': state.epoch}
                ops.save_checkpoint(
                    filename, model, criterions, optimizers, meta)
                logger.info('Checkpoint saved at {}'.format(filename))

            # log training metrics over the complete train_loader
            metrics = train_metric.compute()
            metrics = ops.reduce_dict(metrics, reduction='mean')
            if rank == 0:
                # log scalars
                for k, v in metrics.items():
                    k = 'train/epoch_' + k
                    logger.add_scalar(k, v, global_step=state.epoch)

                # log text message
                msg = 'Epoch [{}/{}] completed!\n Loss: {:.3f}'.format(
                    state.epoch, cfg.num_epochs,
                    float(metrics.pop('loss')))
                # append other available metrics
                for k, v in metrics.items():
                    msg += ' {}: {:.3f}'.format(k, float(v))
                logger.info(msg)

            # run validation loop at frequency
            if val_loader is not None and (
                state.epoch % cfg.val_frequency == 0 or \
                state.epoch == cfg.num_epochs):
                # reset validation metric
                val_metric.reset()

                # run validation loop
                logger.info('Start validation...')
                for val_step, batch in enumerate(val_loader, 1):
                    model.eval()
                    state.val_step = val_step

                    # run a step of forward pass (model.val_step)
                    batch = ops.to_device(
                        batch, device, cfg.non_blocking)
                    with torch.no_grad():
                        output = model.val_step(batch, state)

                    if distributed:
                        outputs = ops.gather(output, dst=0)
                    elif isinstance(model, DataParallel):
                        outputs = output.split(1, dim=0)
                    else:
                        outputs = [output]

                    # log validation metrics
                    if rank == 0:
                        # compute average metrics over processes/threads
                        all_metrics = [val_metric.update(o)
                                       for o in outputs]
                        if all([m is not None for m in all_metrics]):
                            metrics = {
                                k: sum([m[k] for m in all_metrics])
                                for k in all_metrics[0]}
                            metrics = {
                                k: v / len(outputs)
                                for k, v in metrics.items()}
                        else:
                            metrics = None

                        # log validation metrics
                        if val_step == 1 or \
                            val_step % cfg.echo_frequency == 0 or \
                            val_step == len(val_loader):
                            msg = 'Val step: [{}/{}]'.format(
                                val_step, len(val_loader))
                            if metrics is not None:
                                for k, v in metrics.items():
                                    msg += ' {}: {:.3f}'.format(
                                        k, float(v))
                            logger.info(msg)

                # log validation metrics over the complete val_loader
                if rank == 0:
                    metrics = val_metric.compute()
                    if metrics is not None:
                        # log scalars
                        for k, v in metrics.items():
                            k = 'val/epoch_' + k
                            logger.add_scalar(
                                k, v, global_step=state.epoch)

                        # log text message
                        msg = 'Validation on epoch [{}/{}] ' \
                              'completed!\n'.format(
                                  state.epoch, cfg.num_epochs)
                        for k, v in metrics.items():
                            msg += ' {}: {:.3f}'.format(k, float(v))
                        logger.info(msg)
                    else:
                        logger.warning('Got None type output metrics')
                    
                    # check if achieving higher performance
                    if state.best_indicator is not None and \
                        state.best_indicator in metrics and \
                        state.best_value < metrics[state.best_indicator]:
                        # update the best_value and log message
                        state.best_value = metrics[state.best_indicator]
                        logger.info('Recorded higher {}: {:.3f}'.format(
                            state.best_indicator, state.best_value))
                        # save checkpoint
                        filename = osp.join(cfg.log_dir, 'best.pth')
                        meta = {'epoch': state.epoch,
                                state.best_indicator: state.best_value}
                        ops.save_checkpoint(
                            filename, model, criterions, optimizers, meta)
                        logger.info(
                            'Checkpoint saved at {}'.format(filename))

            # reset training metric
            train_metric.reset()

            # update epoch and step
            state.epoch += 1
            state.step = 1
        else:
            state.step += 1

    # destroy distributed process groups
    if distributed:
        ops.cleanup_dist()
