import importlib
from typing import Optional, Union, Any, List, Callable

deepspeed_is_installed = importlib.util.find_spec("deepspeed") is not None
if deepspeed_is_installed:
    import deepspeed

import numpy as np
from scipy.spatial.transform import Rotation
import torch
import torch.nn as nn
import torch.utils.checkpoint

from src.metrics.rmsd import align_pred_to_true

BLOCK_ARG = Any
BLOCK_ARGS = List[BLOCK_ARG]


def get_checkpoint_fn():
    deepspeed_is_configured = (
        deepspeed_is_installed and deepspeed.checkpointing.is_configured()
    )
    if deepspeed_is_configured:
        checkpoint = deepspeed.checkpointing.checkpoint
    else:
        checkpoint = torch.utils.checkpoint.checkpoint

    return checkpoint



@torch.jit.ignore
def checkpoint_blocks(
    blocks: List[Callable],
    args: BLOCK_ARGS,
    blocks_per_ckpt: Optional[int],
) -> BLOCK_ARGS:
    """
    Chunk a list of blocks and run each chunk with activation
    checkpointing. We define a "block" as a callable whose only inputs are
    the outputs of the previous block.

    Implements Subsection 1.11.8

    Args:
        blocks:
            List of blocks
        args:
            Tuple of arguments for the first block.
        blocks_per_ckpt:
            Size of each chunk. A higher value corresponds to fewer
            checkpoints, and trades memory for speed. If None, no checkpointing
            is performed.
    Returns:
        The output of the final block
    """

    def wrap(a):
        return (a,) if type(a) is not tuple else a

    def exec(b, a):
        for block in b:
            a = wrap(block(*a))
        return a

    def chunker(s, e):
        def exec_sliced(*a):
            return exec(blocks[s:e], a)

        return exec_sliced

    # Avoids mishaps when the blocks take just one argument
    args = wrap(args)

    if blocks_per_ckpt is None or not torch.is_grad_enabled():
        return exec(blocks, args)
    elif blocks_per_ckpt < 1 or blocks_per_ckpt > len(blocks):
        raise ValueError("blocks_per_ckpt must be between 1 and len(blocks)")

    checkpoint = get_checkpoint_fn()

    for s in range(0, len(blocks), blocks_per_ckpt):
        e = s + blocks_per_ckpt
        args = checkpoint(chunker(s, e), *args)
        args = wrap(args)

    return args


def centre_random_augmentation(
    x_input_coords: torch.Tensor,
    N_sample: int = 1,
    s_trans: float = 1.0,
    centre_only: bool = False,
    mask: torch.Tensor = None,
    eps: float = 1e-12,
    ref_coords: Optional[torch.Tensor] = None,
):
    """Implements Algorithm 19 in AF3

    Args:
        x_input_coords (torch.Tensor): input coords
            [..., N_atom, 3]
        N_sample (int, optional): the total number of augmentation. Defaults to 1.
        s_trans (float, optional): scale factor of trans. Defaults to 1.0.
        centre_only (bool, optional): if set true, will only perform centering without applying random translation and rotation.
        mask (torch.Tensor, optional): masking for the coords
            [..., N_atom]
        eps (float, optional): small number used for masked mean
    Returns:
        torch.Tensor:  the Augmentation version of input coords
            [..., N_sample, N_atom, 3]
    """

    N_atom = x_input_coords.size(-2)
    device = x_input_coords.device

    # Move to origin [..., N_atom, 3]
    if mask is None:
        x_input_coords = x_input_coords - torch.mean(
            input=x_input_coords, dim=-2, keepdim=True
        )
    else:
        center = (x_input_coords * mask.unsqueeze(dim=-1)).sum(dim=-2) / (
            mask.sum(dim=-1, keepdim=True) + eps
        )
        x_input_coords = x_input_coords - center.unsqueeze(dim=-2)

    # Expand to [..., N_sample, N_atom, 3]
    x_input_coords = expand_at_dim(x_input_coords, dim=-3, n=N_sample)

    if centre_only:
        return x_input_coords

    # N_augment = batch_size * N_sample
    N_augment = torch.numel(x_input_coords[..., 0, 0])

    # Generate N_augment (rot, trans) pairs
    batch_size_shape = x_input_coords.shape[:-3]
    rot_matrix_random = (
        uniform_random_rotation(N_sample=N_augment)
        .to(device)
        .reshape(*batch_size_shape, N_sample, 3, 3)
    ).detach()  # [..., N_sample, 3, 3]
    trans_random = s_trans * torch.randn(size=(*batch_size_shape, N_sample, 3)).to(
        device
    )  # [..., N_sample, 3]
    x_augment_coords = (
        rot_vec_mul(
            r=expand_at_dim(rot_matrix_random, dim=-3, n=N_atom), t=x_input_coords
        )
        + trans_random[..., None, :]
    )  # [..., N_sample, N_atom, 3]

    if ref_coords is not None:
        # align ref_coords to x_augment_coords
        ref_coords = expand_at_dim(ref_coords, dim=-3, n=N_sample)
        ref_coords, _, _ = align_pred_to_true(
            pred_pose=ref_coords,
            true_pose=x_augment_coords,
            atom_mask=None,
        )
        return x_augment_coords, ref_coords

    return x_augment_coords


# Comment: Rotation.random is not supported by torch.compile()
def uniform_random_rotation(N_sample: int = 1) -> torch.Tensor:
    """Generate random rotation matrices with scipy.spatial.transform.Rotation

    Args:
        N_sample (int, optional): the total number of augmentation. Defaults to 1.

    Returns:
        torch.Tensor: N_sample rot metrics
            [N_sample, 3, 3]
    """
    rotation = Rotation.random(num=N_sample)
    rot_matrix = torch.from_numpy(rotation.as_matrix()).float()  # [N_sample, 3, 3]
    return rot_matrix


# this is from openfold.utils.rigid_utils import rot_vec_mul
def rot_vec_mul(r: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
    """Apply rot matrix to vector
    Applies a rotation to a vector. Written out by hand to avoid transfer
    to avoid AMP downcasting.

    Args:
        r (torch.Tensor): the rotation matrices
            [..., 3, 3]
        t (torch.Tensor): the coordinate tensors
            [..., 3]

    Returns:
        torch.Tensor: the rotated coordinates
    """
    x, y, z = torch.unbind(input=t, dim=-1)
    return torch.stack(
        tensors=[
            r[..., 0, 0] * x + r[..., 0, 1] * y + r[..., 0, 2] * z,
            r[..., 1, 0] * x + r[..., 1, 1] * y + r[..., 1, 2] * z,
            r[..., 2, 0] * x + r[..., 2, 1] * y + r[..., 2, 2] * z,
        ],
        dim=-1,
    )


# from openfold.utils.tensor_utils.permute_final_dims
# from openfold.utils.tensor_utils.flatten_final_dims
def permute_final_dims(tensor: torch.Tensor, inds: list[int]) -> torch.Tensor:
    """Permute final dims of tensor

    Args:
        tensor (torch.Tensor): the input tensor
            [...]
        inds (List[int]): the dim to permute

    Returns:
        torch.Tensor: the permuted tensor
    """
    zero_index = -1 * len(inds)
    first_inds = list(range(len(tensor.shape[:zero_index])))
    return tensor.permute(first_inds + [zero_index + i for i in inds])


def flatten_final_dims(t: torch.Tensor, num_dims: int) -> torch.Tensor:
    """Flatten final dims of tensor

    Args:
        t (torch.Tensor): the input tensor
            [...]
        num_dims (int): the number of final dims to flatten

    Returns:
        torch.Tensor: the flattened tensor
    """
    return t.reshape(shape=t.shape[:-num_dims] + (-1,))


def one_hot(x: torch.Tensor, v_bins: torch.Tensor) -> torch.Tensor:
    """Get one hot embedding of x from v_bins

    Args:
        x (torch.Tensor): the input x
            [...]
        v_bins (torch.Tensor): the bin
            [bins]
    Returns:
        torch.Tensor: the one hot embedding of x from v_bins
            [..., bins]
    """
    reshaped_bins = v_bins.view(((1,) * len(x.shape)) + (len(v_bins),))
    diffs = x[..., None] - reshaped_bins
    am = torch.argmin(torch.abs(diffs), dim=-1)
    return nn.functional.one_hot(am, num_classes=len(v_bins)).float()


# this is mostly from openfold.utils.torch_utils import batched_gather
def batched_gather(
    data: torch.Tensor, inds: torch.Tensor, dim: int = 0, no_batch_dims: int = 0
) -> torch.Tensor:
    """Gather data according to indices specify by inds

    Args:
        data (torch.Tensor): the input data
            [..., K, ...]
        inds (torch.Tensor): the indices for gathering data
            [..., N]
        dim (int, optional): along which dimension to gather data by inds (the dim of "K" "N"). Defaults to 0.
        no_batch_dims (int, optional): length of dimensions before the "dim" dimension. Defaults to 0.

    Returns:
        torch.Tensor: gathered data
            [..., N, ...]
    """

    # for the naive case
    if len(inds.shape) == 1 and no_batch_dims == 0 and dim == 0:
        return data[inds]

    ranges = []
    for i, s in enumerate(data.shape[:no_batch_dims]):
        r = torch.arange(s)
        r = r.view(*(*((1,) * i), -1, *((1,) * (len(inds.shape) - i - 1))))
        ranges.append(r)

    remaining_dims = [slice(None) for _ in range(len(data.shape) - no_batch_dims)]
    remaining_dims[dim - no_batch_dims if dim >= 0 else dim] = inds
    ranges.extend(remaining_dims)
    return data[ranges]


def broadcast(src: torch.Tensor, other: torch.Tensor, dim: int) -> torch.Tensor:
    if dim < 0:
        dim = other.dim() + dim
    if src.dim() == 1:
        for _ in range(0, dim):
            src = src.unsqueeze(0)
    for _ in range(src.dim(), other.dim()):
        src = src.unsqueeze(-1)
    src = src.expand(other.size())
    return src


def scatter_sum(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    index = broadcast(index, src, dim)
    if out is None:
        size = list(src.size())
        if dim_size is not None:
            size[dim] = dim_size
        elif index.numel() == 0:
            size[dim] = 0
        else:
            size[dim] = int(index.max()) + 1
        out = torch.zeros(size, dtype=src.dtype, device=src.device)
        return out.scatter_add_(dim, index, src)
    else:
        return out.scatter_add_(dim, index, src)


def scatter_add(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return scatter_sum(src, index, dim, out, dim_size)


def scatter_mul(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    return torch.ops.torch_scatter.scatter_mul(src, index, dim, out, dim_size)


def scatter_mean(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> torch.Tensor:
    out = scatter_sum(src, index, dim, out, dim_size)
    dim_size = out.size(dim)

    index_dim = dim
    if index_dim < 0:
        index_dim = index_dim + src.dim()
    if index.dim() <= index_dim:
        index_dim = index.dim() - 1

    ones = torch.ones(index.size(), dtype=src.dtype, device=src.device)
    count = scatter_sum(ones, index, index_dim, None, dim_size)
    count[count < 1] = 1
    count = broadcast(count, out, dim)
    if out.is_floating_point():
        out.true_divide_(count)
    else:
        out.div_(count, rounding_mode="floor")
    return out


def scatter_min(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_min(src, index, dim, out, dim_size)


def scatter_max(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    return torch.ops.torch_scatter.scatter_max(src, index, dim, out, dim_size)


def scatter(
    src: torch.Tensor,
    index: torch.Tensor,
    dim: int = -1,
    out: Optional[torch.Tensor] = None,
    dim_size: Optional[int] = None,
    reduce: str = "sum",
) -> torch.Tensor:
    r"""
    |

    .. image:: https://raw.githubusercontent.com/rusty1s/pytorch_scatter/
            master/docs/source/_figures/add.svg?sanitize=true
        :align: center
        :width: 400px

    |

    Reduces all values from the :attr:`src` tensor into :attr:`out` at the
    indices specified in the :attr:`index` tensor along a given axis
    :attr:`dim`.
    For each value in :attr:`src`, its output index is specified by its index
    in :attr:`src` for dimensions outside of :attr:`dim` and by the
    corresponding value in :attr:`index` for dimension :attr:`dim`.
    The applied reduction is defined via the :attr:`reduce` argument.

    Formally, if :attr:`src` and :attr:`index` are :math:`n`-dimensional
    tensors with size :math:`(x_0, ..., x_{i-1}, x_i, x_{i+1}, ..., x_{n-1})`
    and :attr:`dim` = `i`, then :attr:`out` must be an :math:`n`-dimensional
    tensor with size :math:`(x_0, ..., x_{i-1}, y, x_{i+1}, ..., x_{n-1})`.
    Moreover, the values of :attr:`index` must be between :math:`0` and
    :math:`y - 1`, although no specific ordering of indices is required.
    The :attr:`index` tensor supports broadcasting in case its dimensions do
    not match with :attr:`src`.

    For one-dimensional tensors with :obj:`reduce="sum"`, the operation
    computes

    .. math::
        \mathrm{out}_i = \mathrm{out}_i + \sum_j~\mathrm{src}_j

    where :math:`\sum_j` is over :math:`j` such that
    :math:`\mathrm{index}_j = i`.

    .. note::

        This operation is implemented via atomic operations on the GPU and is
        therefore **non-deterministic** since the order of parallel operations
        to the same value is undetermined.
        For floating-point variables, this results in a source of variance in
        the result.

    :param src: The source tensor.
    :param index: The indices of elements to scatter.
    :param dim: The axis along which to index. (default: :obj:`-1`)
    :param out: The destination tensor.
    :param dim_size: If :attr:`out` is not given, automatically create output
        with size :attr:`dim_size` at dimension :attr:`dim`.
        If :attr:`dim_size` is not given, a minimal sized output tensor
        according to :obj:`index.max() + 1` is returned.
    :param reduce: The reduce operation (:obj:`"sum"`, :obj:`"mul"`,
        :obj:`"mean"`, :obj:`"min"` or :obj:`"max"`). (default: :obj:`"sum"`)

    :rtype: :class:`Tensor`

    .. code-block:: python

        from torch_scatter import scatter

        src = torch.randn(10, 6, 64)
        index = torch.tensor([0, 1, 0, 1, 2, 1])

        # Broadcasting in the first and last dim.
        out = scatter(src, index, dim=1, reduce="sum")

        print(out.size())

    .. code-block::

        torch.Size([10, 3, 64])
    """
    if reduce == "sum" or reduce == "add":
        return scatter_sum(src, index, dim, out, dim_size)
    if reduce == "mul":
        return scatter_mul(src, index, dim, out, dim_size)
    elif reduce == "mean":
        return scatter_mean(src, index, dim, out, dim_size)
    elif reduce == "min":
        return scatter_min(src, index, dim, out, dim_size)[0]
    elif reduce == "max":
        return scatter_max(src, index, dim, out, dim_size)[0]
    else:
        raise ValueError


def broadcast_token_to_atom(
    x_token: torch.Tensor, atom_to_token_idx: torch.Tensor
) -> torch.Tensor:
    """Broadcast token-level embeddings to atom-level embeddings

    Args:
        x_token (torch.Tensor): token embedding
            [..., N_token, d]
        atom_to_token_idx (torch.Tensor): map atom idx to token idx
            [..., N_atom] or [N_atom]

    Returns:
        torch.Tensor: atom embedding
            [..., N_atom, d]
    """

    if len(atom_to_token_idx.shape) == 1:
        # shape = [N_atom], easy index
        return x_token[..., atom_to_token_idx, :]
    else:
        assert atom_to_token_idx.shape[:-1] == x_token.shape[:-2]

    return batched_gather(
        data=x_token,
        inds=atom_to_token_idx,
        dim=-2,
        no_batch_dims=len(x_token.shape[:-2]),
    )


def aggregate_atom_to_token(
    x_atom: torch.Tensor,
    atom_to_token_idx: torch.Tensor,
    n_token: Optional[int] = None,
    reduce: str = "mean",
) -> torch.Tensor:
    """Aggregate atom embedding to obtain token embedding

    Args:
        x_atom (torch.Tensor): atom-level embedding
            [..., N_atom, d]
        atom_to_token_idx (torch.Tensor): map atom to token idx
            [..., N_atom] or [N_atom]
        n_token (int, optional): number of tokens in total. Defaults to None.
        reduce (str, optional): aggregation method. Defaults to "mean".

    Returns:
        torch.Tensor: token-level embedding
            [..., N_token, d]
    """

    # Broadcasting in the given dim.
    out = scatter(
        src=x_atom, index=atom_to_token_idx, dim=-2, dim_size=n_token, reduce=reduce
    )

    return out


def sample_indices(
    n: int,
    device: torch.device = torch.device("cpu"),
    lower_bound=1,
    strategy: str = "random",
) -> torch.Tensor:
    """Sample msa indices k from uniform[1,n]

    Args:
        n (int): the msa num
        strategy (str): the strategy to sample msa index, random or topk

    Returns:
        torch.Tensor: the sampled indices k
    """
    assert strategy in ["random", "topk"]
    sample_size = torch.randint(low=min(lower_bound, n), high=n + 1, size=(1,)).item()
    if strategy == "random":
        indices = torch.randperm(n=n, device=device)[:sample_size]
    if strategy == "topk":
        indices = torch.arange(sample_size, device=device)
    return indices


def sample_msa_feature_dict_random_without_replacement(
    feat_dict: dict[str, torch.Tensor],
    dim_dict: dict[str, int],
    cutoff: int = 512,
    lower_bound: int = 1,
    strategy: str = "random",
) -> dict[str, torch.Tensor]:
    """Sample a dict of MSA features randomly without replacement.

    Args:
        feat_dict (dict[str, torch.Tensor]): A dict containing the MSA features.
        dim_dict (dict[str, int]): A dict containing the dimensions of the MSA features.
        cutoff (int): The maximum number of features to sample.
        lower_bound (int): The minimum number of features to sample.
        strategy (str): The sampling strategy to use. Can be either "random" or "sequential".

    Returns:
        dict[str, torch.Tensor]: A dict containing the sampled MSA features.
    """
    msa_len = feat_dict["msa"].size(dim=dim_dict["msa"])
    indices = sample_indices(
        n=msa_len,
        device=feat_dict["msa"].device,
        lower_bound=lower_bound,
        strategy=strategy,
    )
    if cutoff > 0:
        indices = indices[:cutoff]

    msa_feat_dict = {
        feat_name: torch.index_select(
            input=feat_dict[feat_name], dim=dim, index=indices
        )
        for feat_name, dim in dim_dict.items()
    }
    return msa_feat_dict


def expand_at_dim(x: torch.Tensor, dim: int, n: int) -> torch.Tensor:
    """expand a tensor at specific dim by n times

    Args:
        x (torch.Tensor): input
        dim (int): dimension to expand
        n (int): expand size

    Returns:
        torch.Tensor: expanded tensor of shape [..., n, ...]
    """
    x = x.unsqueeze(dim=dim)
    if dim < 0:
        dim = x.dim() + dim
    before_shape = x.shape[:dim]
    after_shape = x.shape[dim + 1 :]
    return x.expand(*before_shape, n, *after_shape)


def pad_at_dim(
    x: torch.Tensor,
    dim: int,
    pad_length: Union[tuple[int], list[int]],
    value: float = 0,
) -> torch.Tensor:
    """pad to input x at dimension dim with length pad_length[0] to the left and and pad_length[1] to the right.

    Args:
        x (torch.Tensor): input
        dim (int): padding dimension
        pad_length (Union[Tuple[int], List[int]]): length to pad to the beginning and end.

    Returns:
        torch.Tensor: padded tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    pad = (pad_length[0], pad_length[1])
    if pad == (0, 0):
        return x
    k = n_dim - (dim + 1)
    if k > 0:
        pad_skip = (0, 0) * k
        pad = (*pad_skip, *pad)
    return nn.functional.pad(x, pad=pad, value=value)


def reshape_at_dim(
    x: torch.Tensor, dim: int, target_shape: Union[tuple[int], list[int]]
) -> torch.Tensor:
    """reshape dimension dim of x to target_shape

    Args:
        x (torch.Tensor): input
        dim (int): dimension to reshape
        target_shape (Union[Tuple[int], List[int]]): target_shape of dim

    Returns:
        torch.Tensor: reshaped tensor
    """
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim

    target_shape = tuple(target_shape)
    target_shape = (*x.shape[:dim], *target_shape)
    if dim + 1 < n_dim:
        target_shape = (*target_shape, *x.shape[dim + 1 :])
    return x.reshape(target_shape)


def move_final_dim_to_dim(x: torch.Tensor, dim: int) -> torch.Tensor:
    """
    Move the final dimension of a tensor to a specified dimension.

    Args:
        x (torch.Tensor): Input tensor.
        dim (int): Target dimension to move the final dimension to.

    Returns:
        torch.Tensor: Tensor with the final dimension moved to the specified dimension.
    """
    # permute_final_dims
    n_dim = len(x.shape)
    if dim < 0:
        dim = n_dim + dim
    if dim >= n_dim - 1:
        return x

    new_order = (n_dim - 1,)
    if dim > 0:
        new_order = tuple(range(dim)) + new_order
    if dim < n_dim - 1:
        new_order = new_order + tuple(range(dim, n_dim - 1))

    return x.permute(new_order)


def simple_merge_dict_list(dict_list: list[dict]) -> dict:
    """
    Merge a list of dictionaries into a single dictionary.

    Args:
        dict_list (list[dict]): List of dictionaries to merge.

    Returns:
        dict: Merged dictionary where values are concatenated arrays.
    """
    merged_dict = {}

    def add(key, value):
        merged_dict.setdefault(key, [])
        if isinstance(value, (float, int)):
            value = np.array([value])
        elif isinstance(value, torch.Tensor):
            if value.dim() == 0:
                value = np.array([value.item()])
            else:
                value = value.detach().cpu().numpy()
        elif isinstance(value, np.ndarray):
            pass
        else:
            raise ValueError(f"Unsupported type for metric data: {type(value)}")
        merged_dict[key].append(value)

    for x in dict_list:
        for k, v in x.items():
            add(k, v)
    for k, v in merged_dict.items():
        merged_dict[k] = np.concatenate(v)
    return merged_dict
