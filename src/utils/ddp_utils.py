import os
import random

import numpy as np
import torch


def distributed_available() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


class DistWrapper:
    def __init__(self) -> None:
        self.rank = int(os.environ.get("RANK", 0))
        self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        self.local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
        self.world_size = int(os.environ.get("WORLD_SIZE", 1))
        self.num_nodes = int(self.world_size // self.local_world_size)
        self.node_rank = int(self.rank // self.local_world_size)

    def all_gather_object(self, obj, group=None):
        """Function to gather objects from several distributed processes.
        It is now only used by sync metrics in logger due to security reason.
        """
        if self.world_size > 1 and distributed_available():
            with torch.no_grad():
                obj_list = [None for _ in range(self.world_size)]
                torch.distributed.all_gather_object(obj_list, obj, group=group)
                return obj_list
        else:
            return [obj]


DIST_WRAPPER = DistWrapper()


def seed_everything(seed, deterministic):
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.benchmark = False
        # torch.backends.cudnn.deterministic=True applies to CUDA convolution operations, and nothing else.
        torch.backends.cudnn.deterministic = True
        # torch.use_deterministic_algorithms(True) affects all the normally-nondeterministic operations listed here https://pytorch.org/docs/stable/generated/torch.use_deterministic_algorithms.html?highlight=use_deterministic#torch.use_deterministic_algorithms
        torch.use_deterministic_algorithms(True)
        # https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
