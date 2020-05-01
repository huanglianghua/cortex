# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
"""
This file contains primitives for multi-gpu communication.
This is useful when doing distributed training.
"""

import socket
import torch
import torch.multiprocessing as mp
import torch.distributed as dist
import logging
import functools
import pickle
import numpy as np
from collections import OrderedDict


_LOCAL_PROCESS_GROUP = None
"""
A torch process group which only includes processes that on the same machine as the current process.
This variable is set when processes are spawned by `launch()` in "engine/launch.py".
"""


def get_world_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size()
    return 1


def get_rank():
    if dist.is_available() and dist.is_initialized():
        return dist.get_rank()
    return 0


def get_local_size():
    if dist.is_available() and dist.is_initialized():
        return dist.get_world_size(group=_LOCAL_PROCESS_GROUP)
    return 1


def get_local_rank():
    if dist.is_available() and dist.is_initialized():
        assert _LOCAL_PROCESS_GROUP is not None
        return dist.get_rank(group=_LOCAL_PROCESS_GROUP)
    return 0


def cleanup_dist():
    if dist.is_available() and dist.is_initialized():
        dist.destroy_process_group()


def is_main_process():
    return get_rank() == 0


def synchronize():
    if dist.is_available() and dist.is_initialized() and \
        dist.get_world_size() > 1:
        dist.barrier()
    return


def reduce_dict(input_dict, reduction='mean'):
    assert reduction in ['mean', 'sum']
    world_size = get_world_size()
    if world_size == 1:
        return input_dict
    with torch.no_grad():
        # ensure that the orders of keys are consistent across processes
        if isinstance(input_dict, OrderedDict):
            keys = list(input_dict.keys())
        else:
            keys = sorted(input_dict.keys())
        vals = [input_dict[key] for key in keys]
        vals = torch.stack(vals, dim=0)
        dist.reduce(vals, dst=0)
        if dist.get_rank() == 0 and reduction == 'mean':
            vals /= world_size
        reduced_dict = type(input_dict)([
            (key, val) for key, val in zip(keys, vals)])
    return reduced_dict


@functools.lru_cache()
def _get_global_gloo_group():
    backend = dist.get_backend()
    assert backend in ['gloo', 'nccl']
    if backend == 'nccl':
        return dist.new_group(backend='gloo')
    else:
        return dist.group.WORLD


def _serialize_to_tensor(data, group):
    backend = dist.get_backend(group)
    assert backend in ['gloo', 'nccl']
    device = torch.device('cpu' if backend == 'gloo' else 'cuda')

    buffer = pickle.dumps(data)
    if len(buffer) > 1024 ** 3:
        logger = logging.getLogger(__name__)
        logger.warning(
            'Rank {} trying to all-gather {:.2f} GB of data on device'
            '{}'.format(get_rank(), len(buffer) / (1024 ** 3), device))
    storage = torch.ByteStorage.from_buffer(buffer)
    tensor = torch.ByteTensor(storage).to(device=device)
    return tensor


def _pad_to_largest_tensor(tensor, group):
    world_size = dist.get_world_size(group=group)
    assert world_size >= 1, \
        'gather/all_gather must be called from ranks within' \
        'the give group!'
    local_size = torch.tensor(
        [tensor.numel()], dtype=torch.int64, device=tensor.device)
    size_list = [torch.zeros(
        [1], dtype=torch.int64, device=tensor.device)
        for _ in range(world_size)]
    # gather tensors and compute the maximum size
    dist.all_gather(size_list, local_size, group=group)
    size_list = [int(size.item()) for size in size_list]
    max_size = max(size_list)

    if local_size != max_size:
        padding = torch.zeros(
            (max_size - local_size, ),
            dtype=torch.uint8, device=tensor.device)
        tensor = torch.cat((tensor, padding), dim=0)
    
    return size_list, tensor


def all_gather(data, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group) == 1:
        return [data]
    
    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)
    max_size = max(size_list)

    # receiving tensors from all ranks
    tensor_list = [torch.empty(
        (max_size, ), dtype=torch.uint8, device=tensor.device)
        for _ in size_list]
    dist.all_gather(tensor_list, tensor, group=group)

    data_list = []
    for size, tensor in zip(size_list, tensor_list):
        buffer = tensor.cpu().numpy().tobytes()[:size]
        data_list.append(pickle.loads(buffer))
    
    return data_list


def gather(data, dst=0, group=None):
    if get_world_size() == 1:
        return [data]
    if group is None:
        group = _get_global_gloo_group()
    if dist.get_world_size(group=group) == 1:
        return [data]
    rank = dist.get_rank(group=group)

    tensor = _serialize_to_tensor(data, group)
    size_list, tensor = _pad_to_largest_tensor(tensor, group)

    # receiving tensors from all ranks to dst
    if rank == dst:
        max_size = max(size_list)
        tensor_list = [torch.empty(
            (max_size, ), dtype=torch.uint8, device=tensor.device)
            for _ in size_list]
        dist.gather(tensor, tensor_list, dst=dst, group=group)

        data_list = []
        for size, tensor in zip(size_list, tensor_list):
            buffer = tensor.cpu().numpy().tobytes()[:size]
            data_list.append(pickle.loads(buffer))
        return data_list
    else:
        dist.gather(tensor, [], dst=dst, group=group)
        return []


def shared_random_seed():
    """All workers must call this function, otherwise it will deadblock.
    """
    seed = np.random.randint(2 ** 31)
    all_seeds = all_gather(seed)
    return all_seeds[0]


def _find_free_port():
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # binding to port 0 will cause the OS to find a free port
    sork.bind(('', 0))
    port = sork.getsorckname()[1]
    sock.close()
    # NOTE: there is still a chance the port could be taken by
    # other processes
    return port


def _worker(local_rank, world_size, func, args,
            gpus_per_machine, machine_rank, dist_url):
    assert torch.cuda.is_available(), \
        'CUDA is not available, please check your installation.'
    global_rank = machine_rank * gpus_per_machine + local_rank
    try:
        dist.init_process_group(
            backend='nccl',
            init_method=dist_url,
            world_size=world_size,
            rank=global_rank)
    except Exception as e:
        logger = logging.getLogger(__name__)
        logger.error('Process group URL: {}'.format(dist_url))
        raise e
    # synchronize is needed here to prevent a possible timeout after
    # calling init_process_group.
    synchronize()
    assert gpus_per_machine <= torch.cuda.device_count()
    torch.cuda.set_device(local_rank)

    # setup local process group
    # (which contains ranks within the same machine)
    global _LOCAL_PROCESS_GROUP
    assert _LOCAL_PROCESS_GROUP is None
    num_machines = world_size // gpus_per_machine
    for i in range(num_machines):
        ranks_on_i = list(range(
            i * gpus_per_machine, (i + 1) * gpus_per_machine))
        process_group = dist.new_group(ranks_on_i)
        if i == machine_rank:
            _LOCAL_PROCESS_GROUP = process_group
    
    # call the function
    func(*args)


def launch(func, args, gpus_per_machine, num_machines=1, machine_rank=0,
           dist_url=None):
    world_size = gpus_per_machine * num_machines
    if world_size > 1:
        # determine init_method (dist_url)
        if dist_url == 'auto':
            assert num_machines == 1, \
                'dist_url="auto" cannot work with distributed training.'
            port = _find_free_port()
            dist_url = f'tcp://127.0.0.1:{port}'
        # spawn multi-processing workers
        mp.spawn(
            _worker,
            args=(world_size, func, args,
                  gpus_per_machine, machine_rank, dist_url),
            nprocs=gpus_per_machine,
            daemon=False)
    else:
        func(*args)
