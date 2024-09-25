import torch.utils.checkpoint as checkpoint


def recompute_wrapper(func, *args, is_recompute=True):
    """Function wrapper for recompute"""
    if is_recompute:
        return checkpoint.checkpoint(func, *args)
    else:
        return func(*args)