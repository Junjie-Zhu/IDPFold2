from typing import Dict, List, Literal

import torch
import torch.nn.functional as F
from loguru import logger
from torch import nn
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


class RelativePositionPairFeat(Feature):
    """Computes relative position for residue and chain ids"""

    def __init__(self, rel_pos_dim, r_max, **kwargs):
        super().__init__(dim=rel_pos_dim)
        self.r_max = r_max
        self.LinearNoBias = nn.Linear(2 + 2 * (r_max + 1), rel_pos_dim, bias=False)

    def forward(self, batch):
        if "residue_pdb_idx" not in batch:
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[:2]
            batch["residue_pdb_idx"] = torch.Tensor([[i + 1 for i in range(n)] for _ in range(b)]).to(
                xt.device
            )
        else:
            residue_idx = batch["residue_pdb_idx"]  # [b, n]

        if "chains" not in batch:
            xt = batch["x_t"]  # [b, n, 3]
            b, n = xt.shape[:2]
            batch["chains"] = torch.ones((b, n), dtype=torch.long).to(
                xt.device
            )
        else:
            chain_idx = batch["chains"]  # [b, n]

        same_chain_mask = (chain_idx[:, :, None] == chain_idx[:, None, :]).long()  # [b, n, n]
        rel_pos_chain = F.one_hot(same_chain_mask, 2)  # [b, n, n, 2]

        d_residue = torch.clip(
            input=residue_idx[:, :, None] - residue_idx[:, None, :] + self.r_max,
            min=0, max=2 * self.r_max,
        ) * same_chain_mask + (1 - same_chain_mask) * (2 * self.r_max + 1)  # [b, n, n]
        rel_pos_residue = F.one_hot(d_residue.long(), 2 * (self.r_max + 1))  # [b, n, n, 2 * (r_max + 1)]
        rel_pos = torch.cat([rel_pos_chain, rel_pos_residue], dim=-1)  # [b, n, n, 2 + 2 * (r_max + 1)]
        return self.LinearNoBias(rel_pos.float())  # [b, n, n, rel_pos_dim]


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
        try:
            rpadmask = batch["mask_dict"]["residue_type"]  # [b, n] binary
        except:
            rpadmask = batch["mask"]  # [b, n] binary
        rtype = rtype * rpadmask  # [b, n], the -1 padding becomes 0
        rtype_onehot = F.one_hot(rtype, num_classes=20)  # [b, n, 20]
        rtype_onehot = (
            rtype_onehot * rpadmask[..., None]
        )
        return rtype_onehot * 1.0


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
            elif f == "plm_emb":
                return PLMSeqFeat(**kwargs)
            elif f == "res_type":
                return ResidueTypeSeqFeat(**kwargs)
            else:
                raise IOError(f"Sequence feature {f} not implemented.")

        elif self.mode == "pair":
            if f == "xt_pair_dists":
                return XtPairwiseDistancesPairFeat(**kwargs)
            elif f == "time_emb":
                return TimeEmbeddingPairFeat(**kwargs)
            elif f == "rel_pos":
                return RelativePositionPairFeat(**kwargs)
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