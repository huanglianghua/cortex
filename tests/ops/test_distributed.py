import unittest
import os
import torch
import torch.distributed as dist

import cortex.ops as ops


def _single_process():
    data = [ops.get_rank()] * (ops.get_rank() + 1)
    gathered_data = ops.gather(data, dst=0)
    dict_data = {'avg_rank': torch.Tensor([ops.get_rank()]).cuda()}
    reduced_data = ops.reduce_dict(dict_data, reduction='mean')
    print('world_size:', ops.get_world_size(),
          'rank:', ops.get_rank(),
          'group_size:', ops.get_local_size(),
          'local_rank:', ops.get_local_rank(),
          'is_main_process:', ops.is_main_process(),
          'shared_random_seed:', ops.shared_random_seed(),
          'gathered_data:', gathered_data,
          'reduced_data', reduced_data['avg_rank'].item())
    dist.destroy_process_group()


class TestDistributed(unittest.TestCase):

    def setUp(self):
        os.environ['MASTER_ADDR'] = 'localhost'
        os.environ['MASTER_PORT'] = '8888'
        self.gpus_per_machine = torch.cuda.device_count()
    
    def test_distributed_ops(self):
        ops.launch(_single_process, args=(),
                   gpus_per_machine=self.gpus_per_machine)


if __name__ == '__main__':
    unittest.main()
