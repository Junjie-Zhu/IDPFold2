import inspect
import math
import warnings
from typing import Optional

import torch
import torch.distributed as dist
from torch.optim.lr_scheduler import LRScheduler


def get_adamw(
    model: torch.nn.Module,
    weight_decay: float,
    learning_rate: float,
    betas: tuple[float, float],
    device_type: str,
) -> torch.optim.AdamW:
    """
    Create an AdamW optimizer for the given model with specified parameters.

    Args:
        model (torch.nn.Module): The model for which the optimizer is created.
        weight_decay (float): The weight decay (L2 penalty) for the optimizer.
        learning_rate (float): The learning rate for the optimizer.
        betas (tuple): Coefficients used for computing running averages of gradient and its square.
        device_type (str): The device type ('cuda' or 'cpu') on which the optimizer will operate.

    Returns:
        torch.optim.AdamW: The AdamW optimizer configured with the specified parameters.
    """
    # start with all of the candidate parameters
    param_dict = {pn: p for pn, p in model.named_parameters()}
    # filter out those that do not require grad
    param_dict = {pn: p for pn, p in param_dict.items() if p.requires_grad}
    # create optim groups. Any parameters that is 2D will be weight decayed, otherwise no.
    # i.e. all weight tensors in matmuls + embeddings decay, all biases and layernorms don't.
    decay_params = [p for n, p in param_dict.items() if p.dim() >= 2]
    nodecay_params = [p for n, p in param_dict.items() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    num_decay_params = sum(p.numel() for p in decay_params)
    num_nodecay_params = sum(p.numel() for p in nodecay_params)
    print(
        f"num decayed parameter tensors: {len(decay_params)}, with {num_decay_params:,} parameters"
    )
    print(
        f"num non-decayed parameter tensors: {len(nodecay_params)}, with {num_nodecay_params:,} parameters"
    )
    # Create AdamW optimizer and use the fused version if it is available
    fused_available = "fused" in inspect.signature(torch.optim.AdamW).parameters
    use_fused = fused_available and device_type == "cuda"
    extra_args = dict(fused=True) if use_fused else dict()
    optimizer = torch.optim.AdamW(
        optim_groups, lr=learning_rate, betas=betas, **extra_args
    )
    print(f"using fused AdamW: {use_fused}")

    return optimizer


def get_optimizer(
    model: torch.nn.Module,
    lr: float,
    weight_decay: float = 0.,
    betas: tuple[float, float] = (0.9, 0.999),
    use_adamw: bool = False,
) -> torch.optim.Optimizer:
    if use_adamw:
        optimizer = get_adamw(
            model=model,
            weight_decay=weight_decay,
            learning_rate=lr,
            betas=(betas[0], betas[1]),
            device_type="cuda" if torch.cuda.is_available() else "cpu",
        )
    else:
        optimizer = torch.optim.Adam(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            betas=(betas[0], betas[1]),
        )
    return optimizer


def is_loss_nan_check(loss: torch.Tensor) -> bool:
    """check the validness of the current loss

    Args:
        loss: the loss from the model

    Returns:
        bool: if True, loss is not nan or inf
    """

    def is_nan(x):
        return torch.isnan(x).any() or torch.isinf(x).any()

    def all_reduce_tensor(tensor, op=dist.ReduceOp.SUM):
        if dist.is_initialized():
            dist.all_reduce(tensor, op=op)
        return tensor

    nan_flag = torch.tensor(
        1.0 if is_nan(loss) else 0.0,
        device=loss.device if torch.cuda.is_available() else None,
    )  # support cpu
    # avoid "Watchdog caught collective operation timeout" error
    all_reduce_tensor(nan_flag)
    if nan_flag.item() > 0.0:
        return True
    return False


class CosineAnnealingWithWarmup(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        decay_steps: int,
        lr: float,
        min_lr: float,
        last_epoch: int = -1,
        verbose: bool = False,
    ):
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_steps
        self.lr = lr
        self.min_lr = min_lr
        super().__init__(optimizer, last_epoch, verbose)

    def _get_step_lr(self, step):
        if step <= self.warmup_steps:
            return (step + 1) / (self.warmup_steps + 1) * self.lr
        elif step >= self.decay_steps:
            return self.min_lr
        else:
            decay_ratio = (step - self.warmup_steps) / (
                self.decay_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
            return self.min_lr + coff * (self.lr - self.min_lr)

    def get_lr(self):
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]

    def _get_closed_form_lr(self):
        return [self._get_step_lr(self.last_epoch) for base_lr in self.base_lrs]


# The Alphafold3 Learning Rate Scheduler As in 5.4
class AlphaFold3LRScheduler(LRScheduler):
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        last_epoch: int = -1,
        verbose: bool = False,
        warmup_steps: int = 1000,
        lr: float = 1.8e-3,
        decay_every_n_steps: int = 50000,
        decay_factor: float = 0.95,
    ) -> None:
        self.warmup_steps = warmup_steps
        self.decay_steps = decay_every_n_steps
        self.lr = lr
        self.decay_factor = decay_factor
        super(AlphaFold3LRScheduler, self).__init__(
            optimizer=optimizer, last_epoch=last_epoch, verbose=verbose
        )

    def _get_step_lr(self, step):
        if step <= self.warmup_steps:
            lr = (step + 1) / (self.warmup_steps + 1) * self.lr
        else:
            decay_count = step // self.decay_steps
            lr = self.lr * (self.decay_factor**decay_count)
        return lr

    def get_lr(self) -> list[float]:
        if not self._get_lr_called_within_step:
            warnings.warn(
                "To get the last learning rate computed by the scheduler, "
                "please use `get_last_lr()`.",
                UserWarning,
            )
        return [
            self._get_step_lr(self.last_epoch) for group in self.optimizer.param_groups
        ]


def get_lr_scheduler(
    optimizer: torch.optim.Optimizer,
    lr_scheduler: str = "af3",
    lr: float = 1.8e-3,
    max_steps: int = 500000,
    warmup_steps: int = 1000,
    decay_every_n_steps: int = 50000,
    decay_factor: float = 0.95,
    min_lr: Optional[float] = None,
) -> torch.optim.lr_scheduler.LRScheduler:

    if lr_scheduler == "af3":
        lr_scheduler = AlphaFold3LRScheduler(
            optimizer,
            lr=lr,
            warmup_steps=warmup_steps,
            decay_every_n_steps=decay_every_n_steps,
            decay_factor=decay_factor
        )
    elif lr_scheduler == "cosine_annealing":
        lr_scheduler = CosineAnnealingWithWarmup(
            optimizer,
            warmup_steps=warmup_steps,
            decay_steps=decay_every_n_steps,
            lr=lr,
            min_lr=min_lr,
        )
    elif lr_scheduler == "constant":
        lr_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer,
            factor=1.0,
            total_iters=max_steps,
        )
    else:
        raise ValueError(f"Invalid lr scheduler: [{lr_scheduler}]")
    return lr_scheduler

