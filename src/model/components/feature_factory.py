from typing import Dict, List, Literal

import torch
import torch.nn.functional as F
from loguru import logger
from torch_scatter import scatter_mean

from src.utils.idx_emb_utils import get_index_embedding, get_time_embedding


# ################################
# # # Some auxiliary functions # #
# ################################


def bin_pairwise_distances(x, min_dist, max_dist, dim):
    """
    Takes coordinates and bins the pairwise distances.

    Args:
        x: Coordinates of shape [b, n, 3]
        min_dist: Right limit of first bin
        max_dist: Left limit of last bin
        dim: Dimension of the final one hot vectors

    Returns:
        Tensor of shape [b, n, n, dim] consisting of one-hot vectors
    """
    pair_dists_nm = torch.norm(x[:, :, None, :] - x[:, None, :, :], dim=-1)  # [b, n, n]
    bin_limits = torch.linspace(
        min_dist, max_dist, dim - 1, device=x.device
    )  # Open left and right
    return bin_and_one_hot(pair_dists_nm, bin_limits)  # [b, n, n, pair_dist_dim]


def bin_and_one_hot(tensor, bin_limits):
    """
    Converts a tensor of shape [*] to a tensor of shape [*, d] using the given bin limits.

    Args:
        tensor (Tensor): Input tensor of shape [*]
        bin_limits (Tensor): bin limits [l1, l2, ..., l_{d-1}]. d-1 limits define
            d-2 bins, and the first one is <l1, the last one is >l_{d-1}, giving a total of d bins.

    Returns:
        torch.Tensor: Output tensor of shape [*, d] where d = len(bin_limits) + 1
    """
    bin_indices = torch.bucketize(tensor, bin_limits)
    return torch.nn.functional.one_hot(bin_indices, len(bin_limits) + 1) * 1.0


def indices_force_start_w_one(pdb_idx, mask):
    """
    Takes a tensor with pdb indices for a batch and forces them all to start with the index 1.
    Masked elements are still assigned the index -1.

    Args:
        pdb_idx: tensor of increasing integers (except masked ones fixed to -1), shape [b, n]
        mask: binary tensor, shape [b, n]

    Returns:
        pdb_idx but now all rows start at 1, masked elements are still set to -1.
    """
    first_val = pdb_idx[:, 0][:, None]  # min val is the first one
    pdb_idx = pdb_idx - first_val + 1
    pdb_idx = torch.masked_fill(pdb_idx, ~mask, -1)  # set masked elements to -1
    return pdb_idx


################################
# # Classes for each feature # #
################################


class Feature(torch.nn.Module):
    """Base class for features."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def get_dim(self):
        return self.dim

    def forward(self, batch: Dict):
        pass  # Implemented by each class

    def assert_defaults_allowed(self, batch: Dict, ftype: str):
        """Raises error if default features should not be used to fill-up missing features in the current batch."""
        if "strict_feats" in batch:
            if batch["strict_feats"]:
                raise IOError(
                    f"{ftype} feature requested but no appropriate feature provided. "
                    "Make sure to include the relevant transform in the data config."
                )


class ZeroFeat(Feature):
    """Computes empty feature (zero) of shape [b, n, dim] or [b, n, n, dim],
    depending on sequence or pair features."""

    def __init__(self, dim_feats_out, mode: Literal["seq", "pair"]):
        super().__init__(dim=dim_feats_out)
        self.mode = mode

    def forward(self, batch):
        xt = batch["x_t"]  # [b, n, 3]
        b, n = xt.shape[0], xt.shape[1]
        if self.mode == "seq":
            return torch.zeros((b, n, self.dim), device=xt.device)
        elif self.mode == "pair":
            torch.zeros((b, n, n, self.dim_feats_out), device=xt.device)
        else:
            raise IOError(f"Mode {self.mode} wrong for zero feature")


class TimeEmbeddingSeqFeat(Feature):
    """Computes time embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeEmbeddingPairFeat(Feature):
    """Computes time embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_embedding(t, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, t_emb_dim]


class TimeIntervalEmbeddingSeqFeat(Feature):
    """Computes time interval embedding and returns as sequence feature of shape [b, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        r = batch["r"]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_interval_embedding(t - r, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, :]  # [b, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, t_emb.shape[2]))  # [b, n, t_emb_dim]


class TimeIntervalEmbeddingPairFeat(Feature):
    """Computes time interval embedding and returns as pair feature of shape [b, n, n, t_emb_dim]."""

    def __init__(self, t_emb_dim, **kwargs):
        super().__init__(dim=t_emb_dim)

    def forward(self, batch):
        t = batch["t"]  # [b]
        r = batch["r"]
        xt = batch["x_t"]  # [b, n, 3]
        n = xt.shape[1]
        t_emb = get_time_interval_embedding(t - r, edim=self.dim)  # [b, t_emb_dim]
        t_emb = t_emb[:, None, None, :]  # [b, 1, 1, t_emb_dim]
        return t_emb.expand((t_emb.shape[0], n, n, t_emb.shape[3]))  # [b, n, n, t_emb_dim]


class IdxEmbeddingSeqFeat(Feature):
    """Computes index embedding and returns sequence feature of shape [b, n, idx_emb]."""

    def __init__(self, idx_emb_dim, **kwargs):
        super().__init__(dim=idx_emb_dim)

    def forward(self, batch):
        # If it has the actual residue indices
        if "residue_pdb_idx" in batch:
            inds = batch["residue_pdb_idx"]  # [b, n]
            inds = indices_force_start_w_one(inds, batch["mask"])
        else:
            self.assert_defaults_allowed(batch, "Residue index sequence")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                xt.device
            )  # [b, n]
        return get_index_embedding(inds, edim=self.dim)  # [b, n, idx_embed_dim]


class ChainBreakPerResidueSeqFeat(Feature):
    """Computes a 1D sequence feature indicating if a residue is followed by a chain break, shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        # If it has the actual chain breaks
        if "chain_breaks_per_residue" in batch:
            chain_breaks = batch["chain_breaks_per_residue"] * 1.0  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Chain break sequence")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            chain_breaks = torch.zeros((b, n), device=xt.device) * 1.0  # [b, n]
        return chain_breaks[..., None]  # [b, n, 1]


class XscSeqFeat(Feature):
    """Computes feature from self conditioning coordinates, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        if "x_sc" in batch:
            return batch["x_sc"]  # [b, n, 3]
        else:
            # If we do not provide self-conditioning as input to the nn
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            return torch.zeros(b, n, 3, device=x.device)


class MotifX1SeqFeat(Feature):
    """Computes feature from motif coordinates if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=3)

    def forward(self, batch):
        if "x_motif" in batch:
            return batch["x_motif"]  # [b, n, 3]
        else:
            # If no motif
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            device = x.device
            return torch.zeros(b, n, 3, device=device)


class MotifMaskSeqFeat(Feature):
    """Computes feature from mask of the motif positions if present, seq feature of shape [b, n, 3]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "motif_mask" in batch:
            return batch["motif_mask"].unsqueeze(-1)  # [b, n, 1]
        else:
            # If no motif
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            device = x.device
            return torch.zeros(b, n, device=device).unsqueeze(-1)


class MotifStructureMaskFeat(Feature):
    """Computes feature of the pair wise motif mask of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "fixed_structure_mask" in batch:
            # no need to force 1 since taking difference
            mask = batch["fixed_structure_mask"].unsqueeze(-1)  # [b, n]
        else:
            raise ValueError("No fixed_structure_mask")
        return mask


class MotifX1PairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances for CA backbone atoms of motif atoms and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
            self, x_motif_pair_dist_dim, x_motif_pair_dist_min, x_motif_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_motif_pair_dist_dim)
        self.min_dist = x_motif_pair_dist_min
        self.max_dist = x_motif_pair_dist_max

    def forward(self, batch):
        assert ("x_motif" in batch)
        assert ("fixed_structure_mask" in batch)
        return bin_pairwise_distances(
            x=batch["x_motif"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        ) * batch["fixed_structure_mask"].unsqueeze(-1)  # [b, n, n, pair_dist_dim]


class SequenceSeparationPairFeat(Feature):
    """Computes sequence separation and returns feature of shape [b, n, n, seq_sep_dim]."""

    def __init__(self, seq_sep_dim, **kwargs):
        super().__init__(dim=seq_sep_dim)

    def forward(self, batch):
        if "residue_pdb_idx" in batch:
            # no need to force 1 since taking difference
            inds = batch["residue_pdb_idx"]  # [b, n]
        else:
            self.assert_defaults_allowed(batch, "Relative sequence separation pair")
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            inds = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                xt.device
            )  # [b, n]

        seq_sep = inds[:, :, None] - inds[:, None, :]  # [b, n, n]

        # Dimension should be odd, bins limits [-(dim/2-1), ..., -1.5, -0.5, 0.5, 1.5, ..., dim/2-1]
        # gives dim-2 bins, and the first and last for values beyond the bin limits
        assert (
                self.dim % 2 == 1
        ), "Relative seq separation feature dimension must be odd and > 3"

        # Create bins limits [..., -3.5, -2.5, -1.5, -0.5, 0.5, 1.5, 2.3, 3.5, ...]
        # Equivalent to binning relative sequence separation
        low = -(self.dim / 2.0 - 1)
        high = self.dim / 2.0 - 1
        bin_limits = torch.linspace(low, high, self.dim - 1, device=inds.device)

        return bin_and_one_hot(seq_sep, bin_limits)  # [b, n, n, seq_sep_dim]


class XtPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(self, xt_pair_dist_dim, xt_pair_dist_min, xt_pair_dist_max, **kwargs):
        super().__init__(dim=xt_pair_dist_dim)
        self.min_dist = xt_pair_dist_min
        self.max_dist = xt_pair_dist_max

    def forward(self, batch):
        return bin_pairwise_distances(
            x=batch["x_t"],
            min_dist=self.min_dist,
            max_dist=self.max_dist,
            dim=self.dim,
        )  # [b, n, n, pair_dist_dim]


class XscPairwiseDistancesPairFeat(Feature):
    """Computes pairwise distances and returns feature of shape [b, n, n, dim_pair_dist]."""

    def __init__(
            self, x_sc_pair_dist_dim, x_sc_pair_dist_min, x_sc_pair_dist_max, **kwargs
    ):
        super().__init__(dim=x_sc_pair_dist_dim)
        self.min_dist = x_sc_pair_dist_min
        self.max_dist = x_sc_pair_dist_max

    def forward(self, batch):
        if "x_sc" in batch:
            return bin_pairwise_distances(
                x=batch["x_sc"],
                min_dist=self.min_dist,
                max_dist=self.max_dist,
                dim=self.dim,
            )  # [b, n, n, pair_dist_dim]
        else:
            # If we do not provide self-conditioning as input to the nn
            x = batch["x_t"]
            b, n = x.shape[0], x.shape[1]
            return torch.zeros(b, n, n, self.dim, device=x.device)


class ResidueTypeSeqFeat(Feature):
    """
    Computes feature from residue type, feature of shape [b, n, 20].

    Residue type is an integer in {0, 1, ..., 19}, coorsponding to the 20 aa types.
    Feature is a one-hot vector of dimension 20.

    Note that in residue type the padding is done with a -1, but this function
    multiplies with the mask.
    """

    def __init__(self, **kwargs):
        super().__init__(dim=20)

    def forward(self, batch):
        assert (
            "residue_type" in batch
        ), "`residue_type` not in batch, cannot compute ResidueTypeSeqFeat"
        rtype = batch["residue_type"]  # [b, n]
        rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]
        rtype_onehot = (
            rtype_onehot * rpadmask[..., None]
        )
        return rtype_onehot * 1.0


class ChainIdxSeqFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            mask = batch["chains"].unsqueeze(-1)  # [b, n, 1]
        else:
            raise ValueError("chains")
        return mask


class ChainIdxPairFeat(Feature):
    """Gets chain idx feature (-1 for padding) and returns feature of shape [b, n, n, 1]."""

    def __init__(self, **kwargs):
        super().__init__(dim=1)

    def forward(self, batch):
        if "chains" in batch:
            seq_mask = batch["chains"]  # [b, n]
            mask = torch.einsum("bi,bj->bij", seq_mask, seq_mask).unsqueeze(
                -1
            )  # [b, n, n, 1]
        else:
            raise ValueError("chains")
        return mask


class PLMSeqFeat(Feature):
    """Computes PLM sequence feature, shape [b, n, plm_dim]."""

    def __init__(self, plm_in_dim, plm_out_dim, **kwargs):
        super().__init__(dim=plm_out_dim)

        # self.layernorm = torch.nn.LayerNorm(plm_in_dim)
        self.linear = torch.nn.Linear(plm_in_dim, plm_out_dim)
        self.relu = torch.nn.ReLU()

    def forward(self, batch):
        if "plm_emb" in batch:
            plm_embedding = batch["plm_emb"]
            plm_mask = (torch.sum(plm_embedding, dim=(-1, -2)) != 0).float()  # [b]
            return self.relu(self.linear(plm_embedding)) * plm_mask[..., None, None]  # [b, n, plm_dim]
        else:
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[0], xt.shape[1]
            return torch.zeros(b, n, self.dim, device=xt.device)


####################################
# # Class that produces features # #
####################################


class FeatureFactory(torch.nn.Module):
    def __init__(
            self,
            feats: List[str],
            dim_feats_out: int,
            use_ln_out: bool,
            mode: Literal["seq", "pair"],
            **kwargs,
    ):
        """
        Sequence features include:
            - "res_seq_pdb_idx", requires transform ResidueSequencePositionPdbTransform
            - "time_emb"
            - "chain_break_per_res", requires transform ChainBreakPerResidueTransform
            - "fold_emb"
            - "x_sc"

        Pair features include:
            - "xt_pair_dists"
            - "x_sc_pair_dists"
            - "rel_seq_sep"
            - "time_emb"
        """
        super().__init__()
        self.mode = mode

        self.ret_zero = True if (feats is None or len(feats) == 0) else False
        if self.ret_zero:
            logger.info("No features requested")
            self.zero_creator = ZeroFeat(dim_feats_out=dim_feats_out, mode=mode)
            return

        self.feat_creators = torch.nn.ModuleList(
            [self.get_creator(f, **kwargs) for f in feats]
        )
        self.ln_out = (
            torch.nn.LayerNorm(dim_feats_out) if use_ln_out else torch.nn.Identity()
        )
        self.linear_out = torch.nn.Linear(
            sum([c.get_dim() for c in self.feat_creators]), dim_feats_out, bias=False
        )

    def get_creator(self, f, **kwargs):
        """Returns the right class for the requested feature f (a string)."""

        if self.mode == "seq":
            if f == "time_emb":
                return TimeEmbeddingSeqFeat(**kwargs)
            elif f == "res_seq_pdb_idx":
                return IdxEmbeddingSeqFeat(**kwargs)
            elif f == "chain_break_per_res":
                return ChainBreakPerResidueSeqFeat(**kwargs)
            elif f == "x_sc":
                return XscSeqFeat(**kwargs)
            elif f == "motif_x1":
                return MotifX1SeqFeat(**kwargs)
            elif f == "motif_sequence_mask":
                return MotifMaskSeqFeat(**kwargs)
            elif f == "plm_emb":
                return PLMSeqFeat(**kwargs)
            elif f == "res_type":
                return ResidueTypeSeqFeat(**kwargs)
            elif f == "chain_idx":
                return ChainIdxSeqFeat(**kwargs)
            elif f == "time_interval_emb":
                return TimeIntervalEmbeddingPairFeat(**kwargs)

            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            if f == "xt_pair_dists":
                return XtPairwiseDistancesPairFeat(**kwargs)
            elif f == "x_sc_pair_dists":
                return XscPairwiseDistancesPairFeat(**kwargs)
            elif f == "rel_seq_sep":
                return SequenceSeparationPairFeat(**kwargs)
            elif f == "time_emb":
                return TimeEmbeddingPairFeat(**kwargs)
            elif f == "motif_x1_pair_dists":
                return MotifX1PairwiseDistancesPairFeat(**kwargs)
            elif f == "motif_structure_mask":
                return MotifStructureMaskFeat(**kwargs)
            elif f == "chain_idx_pair":
                return ChainIdxPairFeat(**kwargs)
            elif f == "time_interval_emb":
                return TimeIntervalEmbeddingPairFeat(**kwargs)
            else:
                raise IOError(f"Pair feature {f} not implemented.")

        else:
            raise IOError(
                f"Wrong feature mode (creator): {self.mode}. Should be 'seq' or 'pair'."
            )

    def apply_padding_mask(self, feature_tensor, mask):
        """
        Applies mask to features.

        Args:
            feature_tensor: tensor with requested features, shape [b, n, d] of [b, n, n, d] depending on self.mode ('seq' or 'pair')
            mask: Binary mask, shape [b, n]

        Returns:
            Masked features, same shape as input tensor.
        """
        if self.mode == "seq":
            return feature_tensor * mask[..., None]  # [b, n, d]
        elif self.mode == "pair":
            mask_pair = mask[:, None, :] * mask[:, :, None]  # [b, n, n]
            return feature_tensor * mask_pair[..., None]  # [b, n, n, d]
        else:
            raise IOError(
                f"Wrong feature mode (pad mask): {self.mode}. Should be 'seq' or 'pair'."
            )

    def forward(self, batch):
        """Returns masked features, shape depends on mode, either 'seq' or 'pair'."""
        # If no features requested just return the zero tensor of appropriate dimensions
        if self.ret_zero:
            return self.zero_creator(batch)

        # Compute requested features
        feature_tensors = []
        for fcreator in self.feat_creators:
            feature_tensors.append(
                fcreator(batch)
            )  # [b, n, dim_f] or [b, n, n, dim_f] if seq or pair mode

        # Concatenate features and mask
        features = torch.cat(
            feature_tensors, dim=-1
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        features = self.apply_padding_mask(
            features, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]

        # Linear layer and mask
        features_proc = self.ln_out(
            self.linear_out(features)
        )  # [b, n, dim_f] or [b, n, n, dim_f]
        return self.apply_padding_mask(
            features_proc, batch["mask"]
        )  # [b, n, dim_f] or [b, n, n, dim_f]