from math import prod
from typing import Literal, Any
import copy
from abc import ABC, abstractmethod

import torch
import wget
from loguru import logger
from scipy.spatial.transform import Rotation as Scipy_Rotation
from torch_geometric.data import Data



def sample_uniform_rotation(shape=(), dtype=None, device=None) -> torch.Tensor:
    """Samples rotation matrices uniformly from SO(3).

    Args:
        shape: Batch dimensions for sampling multiple rotations
        dtype: Data type for the output tensor
        device: Device to place the output tensor on

    Returns:
        Tensor of shape [*shape, 3, 3] containing uniformly sampled rotation matrices
    """
    return torch.tensor(
        Scipy_Rotation.random(prod(shape)).as_matrix(),
        device=device,
        dtype=dtype,
    ).reshape(*shape, 3, 3)


class BaseTransform(ABC):
    def __call__(self, data: Any) -> Any:
        # Shallow-copy the data so that we prevent in-place data modification.
        return self.forward(copy.copy(data))

    @abstractmethod
    def forward(self, data: Any) -> Any:
        pass

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}()'


class CopyCoordinatesTransform(BaseTransform):
    """Creates a backup copy of coordinates before applying modifications.

    This transform copies the original coordinates to coords_unmodified before any
    other transformations (like noising or rotations) are applied.
    """

    def forward(self, graph: Data) -> Data:
        """Copies coordinates to coords_unmodified.

        Args:
            graph: PyG Data object containing protein structure data

        Returns:
            Modified graph with coords_unmodified added
        """
        graph.coords_unmodified = graph.coords.clone()
        return graph


class ChainBreakPerResidueTransform(BaseTransform):
    """Identifies chain breaks in protein structures.

    Creates a binary mask indicating whether each residue has a chain break,
    determined by CA-CA distances exceeding a threshold.
    """

    def __init__(self, chain_break_cutoff: float = 4.0):
        """Initializes the transform.

        Args:
            chain_break_cutoff: Maximum allowed distance between consecutive CA atoms
                before considering it a chain break
        """
        self.chain_break_cutoff = chain_break_cutoff

    def forward(self, graph: Data) -> Data:
        """Identifies chain breaks and adds mask to graph.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with added chain_breaks_per_residue mask
        """
        ca_coords = graph.coords[:, 1, :]
        ca_dists = torch.norm(ca_coords[1:] - ca_coords[:-1], dim=1)
        chain_breaks_per_residue = ca_dists > self.chain_break_cutoff
        graph.chain_breaks_per_residue = torch.cat(
            (
                chain_breaks_per_residue,
                torch.tensor([False], dtype=torch.bool, device=chain_breaks_per_residue.device),
            )
        )
        return graph


class PaddingTransform(BaseTransform):
    """Pads tensors in graph to a specified maximum size.

    Ensures all tensors in the graph have consistent size by padding
    with a fill value up to max_size along the first dimension.
    """

    def __init__(self, max_size=256, fill_value=0):
        """Initializes the transform.

        Args:
            max_size: Target size for padding
            fill_value: Value to use for padding
        """
        self.max_size = max_size
        self.fill_value = fill_value

    def forward(self, graph: Data) -> Data:
        """Applies padding to all applicable tensors in graph.

        Args:
            graph: PyG Data object to pad

        Returns:
            Graph with padded tensors
        """
        for key, value in graph:
            if isinstance(value, torch.Tensor):
                if value.dim() >= 1:
                    pad_dim = 0
                    graph[key] = self.pad_tensor(value, self.max_size, pad_dim, self.fill_value)
        return graph

    def pad_tensor(self, tensor, max_size, dim, fill_value=0):
        """Pads a single tensor to specified size.

        Args:
            tensor: Tensor to pad
            max_size: Target size
            dim: Dimension to pad
            fill_value: Value to use for padding

        Returns:
            Padded tensor
        """
        if tensor.size(dim) >= max_size:
            return tensor
        pad_size = max_size - tensor.size(dim)
        padding = [0] * (2 * tensor.dim())
        padding[2 * (tensor.dim() - 1 - dim) + 1] = pad_size
        return torch.nn.functional.pad(tensor, pad=tuple(padding), mode="constant", value=fill_value)

    def __repr__(self) -> str:
        """Get a string representation of the class.

        Returns:
            str: String representation of the class
        """
        return f"{self.__class__.__name__}(max_size={self.max_size}, fill_value={self.fill_value})"


class GlobalRotationTransform(BaseTransform):
    """Applies random global rotation to protein coordinates.

    Should be used as the first transform that modifies coordinates to maintain
    consistency in subsequent transformations.
    """

    def __init__(self, rotation_strategy: Literal["uniform"] = "uniform"):
        """Initializes the transform.

        Args:
            rotation_strategy: Method for sampling rotations. Currently only "uniform" supported
        """
        self.rotation_strategy = rotation_strategy

    def forward(self, graph: Data) -> Data:
        """Applies random rotation to coordinates.

        Args:
            graph: PyG Data object containing protein structure

        Returns:
            Graph with rotated coordinates

        Raises:
            ValueError: If rotation_strategy is not supported
        """
        if self.rotation_strategy == "uniform":
            rot = sample_uniform_rotation(dtype=graph.coords.dtype, device=graph.coords.device)
        else:
            raise ValueError(f"Rotation strategy {self.rotation_strategy} not supported")
        graph.coords = torch.matmul(graph.coords, rot)
        return graph

