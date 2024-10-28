import torch
import torch.nn.functional as F

from src.models.components.diffusion_module import tile_batch_dim


def weighted_rigid_align(
    pred_coords,
    true_coords,
    mask,
    return_transform=False,
):
    """Compute the weighted rigid alignment.

    The check for ambiguous rotation and low rank of cross-correlation between aligned point
    clouds is inspired by
    https://pytorch3d.readthedocs.io/en/latest/_modules/pytorch3d/ops/points_alignment.html.

    :param pred_coords: Predicted coordinates.
    :param true_coords: True coordinates.
    :param mask: The mask for variable lengths.
    :param return_transform: Whether to return the transformation matrix.
    :return: The optimally aligned coordinates.
    """

    batch_size, num_points, dim = pred_coords.shape
    weights = torch.ones_like(pred_coords[..., 0]).unsqueeze(-1)

    if mask is not None:
        pred_coords = pred_coords * mask[..., None]
        true_coords = true_coords * mask[..., None]
        weights = weights * mask[..., None]

    # Compute weighted centroids
    true_centroid = (true_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )
    pred_centroid = (pred_coords * weights).sum(dim=1, keepdim=True) / weights.sum(
        dim=1, keepdim=True
    )

    # Center the coordinates
    true_coords_centered = true_coords - true_centroid
    pred_coords_centered = pred_coords - pred_centroid

    # Compute the weighted covariance matrix
    cov_matrix = torch.einsum(
        "b n i, b n j -> b i j", weights * true_coords_centered, pred_coords_centered
    )

    # Compute the SVD of the covariance matrix
    U, S, V = torch.svd(cov_matrix)
    U_T = U.transpose(-2, -1)

    det = torch.det(torch.einsum("b i j, b j k -> b i k", V, U_T))

    # Ensure proper rotation matrix with determinant 1
    diag = torch.eye(dim, dtype=det.dtype, device=det.device)
    diag = tile_batch_dim(diag, batch_size).view(batch_size, dim, dim)

    diag[:, -1, -1] = det
    rot_matrix = torch.einsum("b i j, b j k, b k l -> b i l", V, diag, U_T)

    # Apply the rotation and translation
    true_aligned_coords = (
        torch.einsum("b i j, b n j -> b n i", rot_matrix, true_coords_centered) + pred_centroid
    )
    true_aligned_coords.detach_()

    if return_transform:
        translation = true_centroid - torch.einsum(
            "b i j, b ... j -> b ... i", rot_matrix, pred_centroid
        )
        return true_aligned_coords, rot_matrix, translation

    return true_aligned_coords


def weighted_MSE_loss(
    pred_coords,
    true_coords,
    weights,
    mask,
):
    """Compute the weighted MSE loss.

    :param pred_coords: Predicted coordinates.
    :param true_coords: True coordinates.
    :param weights: The weights for the loss.
    :param mask: The mask for variable lengths.
    :return: The weighted MSE loss.
    """

    aligned_coords = weighted_rigid_align(pred_coords, true_coords, mask)

    losses = weights.unsqueeze(-1) * F.mse_loss(pred_coords, aligned_coords, reduction = 'none') / 3.
    loss = losses[mask.to(torch.int64)].mean()
    return loss


