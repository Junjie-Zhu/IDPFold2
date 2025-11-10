# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.


from typing import Dict

import einops
import torch

from src.model.components.feature_factory import FeatureFactory
from src.model.components.pair_bias_attn import PairBiasAttention
from src.model.components.af3_modules import (
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)
from src.model.components.moe_modules import MoE


class MultiHeadAttention(torch.nn.Module):
    """Typical multi-head self-attention attention using pytorch's module."""

    def __init__(self, dim_token, nheads, dropout=0.0):
        super().__init__()

        self.to_q = torch.nn.Linear(dim_token, dim_token)
        self.to_kv = torch.nn.Linear(dim_token, 2 * dim_token, bias=False)

        self.mha = torch.nn.MultiheadAttention(
            embed_dim=dim_token,
            num_heads=nheads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, mask):
        """
        Args:
            x: Input sequence, shape [b, n, dim_token]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence, shape [b, n, dim_token]
        """
        query = self.to_q(x)  # [b, n, dim_token]
        key, value = self.to_kv(x).chunk(2, dim=-1)  # Each [b, n, dim_token]
        return (
            self.mha(
                query=query,
                key=key,
                value=value,
                key_padding_mask=~mask,  # Indicated what should be ignores with True, that's why the ~
                need_weights=False,
                is_causal=False,
            )[0]
            * mask[..., None]
        )  # [b, n, dim_token]


class MultiHeadAttentionADALN(torch.nn.Module):
    """Typical multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, nheads, dim_cond, dropout=0.0):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = MultiHeadAttention(
            dim_token=dim_token, nheads=nheads, dropout=dropout
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        x = self.adaln(x, cond, mask)
        x = self.mha(x, mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class MultiHeadBiasedAttentionADALN_MM(torch.nn.Module):
    """Pair biased multi-head self-attention with adaptive layer norm applied to input
    and adaptive scaling applied to output."""

    def __init__(self, dim_token, dim_pair, nheads, dim_cond, use_qkln):
        super().__init__()
        dim_head = int(dim_token // nheads)
        self.adaln = AdaptiveLayerNorm(dim=dim_token, dim_cond=dim_cond)
        self.mha = PairBiasAttention(
            node_dim=dim_token,
            dim_head=dim_head,
            heads=nheads,
            bias=True,
            dim_out=dim_token,
            qkln=use_qkln,
            pair_dim=dim_pair,
        )
        self.scale_output = AdaptiveLayerNormOutputScale(
            dim=dim_token, dim_cond=dim_cond
        )

    def forward(self, x, pair_rep, cond, mask):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: Conditioning variables, shape [b, n, dim_cond]
            pair_rep: Pair represnetation, shape [b, n, n, dim_pair]
            mask: Binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim_token].
        """
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        x = self.adaln(x, cond, mask)
        x = self.mha(node_feats=x, pair_feats=pair_rep, mask=pair_mask)
        x = self.scale_output(x, cond, mask)
        return x * mask[..., None]


class TransitionADALN(torch.nn.Module):
    """Transition layer with adaptive layer norm applied to input and adaptive
    scaling aplied to output."""

    def __init__(self, *, dim, dim_cond, expansion_factor=4):
        super().__init__()
        self.adaln = AdaptiveLayerNorm(dim=dim, dim_cond=dim_cond)
        self.transition = Transition(
            dim=dim, expansion_factor=expansion_factor, layer_norm=False
        )
        self.scale_output = AdaptiveLayerNormOutputScale(dim=dim, dim_cond=dim_cond)

    def forward(self, x, cond, mask, **kwargs):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]

        Returns:
            Updated sequence representation, shape [b, n, dim]
        """
        x = self.adaln(x, cond, mask)  # [b, n, dim]
        x = self.transition(x, mask)  # [b, n, dim]
        x = self.scale_output(x, cond, mask)  # [b, n, dim]
        return x * mask[..., None]  # [b, n, dim]


class MultiheadAttnAndTransition(torch.nn.Module):
    """Layer that applies mha and transition to a sequence representation. Both layers are their adaptive versions
    which rely on conditining variables (see above).

    Args:
        dim_token: Token dimension in sequence representation.
        dim_pair: Dimension of pair representation.
        nheads: Number of attention heads.
        dim_cond: Dimension of conditioning variables.
        residual_mha: Whether to use a residual connection in the mha layer.
        residual_transition: Whether to use a residual connection in the transition layer.
        parallel_mha_transition: Whether to run mha and transition in parallel or sequentially.
        use_attn_pair_bias: Whether to use a pair represnetation to bias attention.
        use_qkln: Whether to use layer norm on keyus and queries for attention.
        dropout: droput use in the self-attention layer.
    """

    def __init__(
        self,
        dim_token,
        dim_pair,
        nheads,
        dim_cond,
        residual_mha,
        residual_transition,
        parallel_mha_transition,
        use_attn_pair_bias,
        use_qkln,
        dropout=0.0,
        expansion_factor=2,
        use_moe=False,
        n_experts=5,
        n_activated_experts=2,
        dim_moe_cond=0,
        capacity_factor=1.25,
        normalize_expert_weights=True,
        load_balance=True,
    ):
        super().__init__()
        self.parallel = parallel_mha_transition
        self.use_attn_pair_bias = use_attn_pair_bias

        # If parallel do not allow both layers to have a residual connection since it leads to adding x twice
        if self.parallel and residual_mha and residual_transition:
            residual_transition = False

        self.residual_mha = residual_mha
        self.residual_transition = residual_transition

        self.mhba = MultiHeadBiasedAttentionADALN_MM(
            dim_token=dim_token,
            dim_pair=dim_pair,
            nheads=nheads,
            dim_cond=dim_cond,
            use_qkln=use_qkln,
        )

        if not use_moe:
            self.transition = TransitionADALN(
                dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
            )
        else:
            single_expert = TransitionADALN(
                dim=dim_token, dim_cond=dim_cond, expansion_factor=expansion_factor
            )
            self.transition = MoE(
                n_experts=n_experts,
                n_activated_experts=n_activated_experts,
                expert=single_expert,
                dim=dim_token,
                dim_router_cond=dim_moe_cond,
                capacity_factor=capacity_factor,
                normalize_expert_weights=normalize_expert_weights,
                load_balance=load_balance,
            )
        self.use_moe = use_moe

    def _apply_mha(self, x, pair_rep, cond, mask):
        x_attn = self.mhba(x, pair_rep, cond, mask)
        if self.residual_mha:
            x_attn = x_attn + x
        return x_attn * mask[..., None]

    def _apply_transition(self, x, cond, mask, force_moe_capacity=True):
        x_tr = self.transition(x, cond, mask, force_moe_capacity)
        if self.residual_transition:
            x_tr = x_tr + x
        return x_tr * mask[..., None]

    def forward(self, x, pair_rep, cond, mask, force_moe_capacity=True):
        """
        Args:
            x: Input sequence representation, shape [b, n, dim_token]
            cond: conditioning variables, shape [b, n, dim_cond]
            mask: binary mask, shape [b, n]
            pair_rep: Pair representation (if provided, if no bias will be ignored), shape [b, n, n, dim_pair] or None

        Returns:
            Updated sequence representation, shape [b, n, dim].
        """
        x = x * mask[..., None]
        if self.parallel:
            x = self._apply_mha(x, pair_rep, cond, mask) + self._apply_transition(
                x, cond, mask, force_moe_capacity
            )
        else:
            x = self._apply_mha(x, pair_rep, cond, mask)
            x = self._apply_transition(x, cond, mask, force_moe_capacity)
        return x * mask[..., None]


class PairReprBuilder(torch.nn.Module):
    """
    Builds initial pair representation. Essentially the pair feature factory, but potentially with
    an adaptive layer norm layer as well.
    """

    def __init__(self, feats_repr, feats_cond, dim_feats_out, dim_cond_pair, **kwargs):
        super().__init__()

        self.init_repr_factory = FeatureFactory(
            feats=feats_repr,
            dim_feats_out=dim_feats_out,
            use_ln_out=True,
            mode="pair",
            **kwargs,
        )

        self.cond_factory = None  # Build a pair feature for conditioning and use it for adaln the pair representation
        if feats_cond is not None:
            if len(feats_cond) > 0:
                self.cond_factory = FeatureFactory(
                    feats=feats_cond,
                    dim_feats_out=dim_cond_pair,
                    use_ln_out=True,
                    mode="pair",
                    **kwargs,
                )
                self.adaln = AdaptiveLayerNorm(
                    dim=dim_feats_out, dim_cond=dim_cond_pair
                )

    def forward(self, batch_nn):
        mask = batch_nn["mask"]  # [b, n]
        pair_mask = mask[:, :, None] * mask[:, None, :]  # [b, n, n]
        repr = self.init_repr_factory(batch_nn)  # [b, n, n, dim_feats_out]
        if self.cond_factory is not None:
            cond = self.cond_factory(batch_nn)  # [b, n, n, dim_cond]
            repr = self.adaln(repr, cond, pair_mask)
        return repr


class ProteinTransformerAF3(torch.nn.Module):
    """
    Final neural network mimicking the one used in AF3 diffusion. It consists of:

    (1) Input preparation
    (1.a) Initial sequence representation from features
    (1.b) Embed coordaintes and add to initial sequence representation
    (1.c) Conditioning variables from features

    (2) Main trunk
    (2.a) A sequence of layers similar to algorithm 23 of AF3 (multi head attn, transition) using adaptive layer norm
    and adaptive output scaling (also from adaptive layer norm paper)

    (3) Recovering 3D coordinates
    (3.a) A layer that takes as input tokens and produces coordinates
    """

    def __init__(self, **kwargs):
        """
        Initializes the NN. The seqs and pair representations used are just zero in case
        no features are required."""
        super(ProteinTransformerAF3, self).__init__()
        self.nlayers = kwargs["nlayers"]
        self.token_dim = kwargs["token_dim"]
        self.pair_repr_dim = kwargs["pair_repr_dim"]
        self.feats_pair_cond = kwargs.get("feats_pair_cond", [])
        self.use_qkln = kwargs.get("use_qkln", False)

        # Registers
        self.num_registers = kwargs.get("num_registers", None)
        if self.num_registers is None or self.num_registers <= 0:
            self.num_registers = 0
            self.registers = None
        else:
            self.num_registers = int(self.num_registers)
            self.registers = torch.nn.Parameter(
                torch.randn(self.num_registers, self.token_dim) / 20.0
            )

        # To encode corrupted 3d positions
        self.linear_3d_embed = torch.nn.Linear(3, kwargs["token_dim"], bias=False)

        # To form initial representation
        self.init_repr_factory = FeatureFactory(
            feats=kwargs["feats_init_seq"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_seq"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr"],
            feats_cond=kwargs["feats_pair_cond"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

        self.n_experts = kwargs["n_experts"]
        self.top_k = kwargs["n_activated_experts"]

        # Trunk layers
        self.transformer_layers = torch.nn.ModuleList(
            [
                MultiheadAttnAndTransition(
                    dim_token=kwargs["token_dim"],
                    dim_pair=kwargs["pair_repr_dim"],
                    nheads=kwargs["nheads"],
                    dim_cond=kwargs["dim_cond"],
                    residual_mha=kwargs["residual_mha"],
                    residual_transition=kwargs["residual_transition"],
                    parallel_mha_transition=kwargs["parallel_mha_transition"],
                    use_attn_pair_bias=kwargs["use_attn_pair_bias"],
                    use_qkln=self.use_qkln,
                    use_moe=kwargs["use_moe"],
                    n_experts=self.n_experts,
                    n_activated_experts=self.top_k,
                    dim_moe_cond=kwargs["dim_moe_cond"],
                    capacity_factor=kwargs["capacity_factor"],
                    normalize_expert_weights=kwargs["normalize_expert_weights"],
                    load_balance=kwargs["training"],
                )
                for _ in range(self.nlayers)
            ]
        )

        self.coors_3d_decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(kwargs["token_dim"]),
            torch.nn.Linear(kwargs["token_dim"], 3, bias=False),
        )

    def _extend_w_registers(self, seqs, pair, mask, cond_seq):
        """
        Extends the sequence representation, pair representation, mask and indices with registers.

        Args:
            - seqs: sequence representation, shape [b, n, dim_token]
            - pair: pair representation, shape [b, n, n, dim_pair]
            - mask: binary mask, shape [b, n]
            - cond_seq: tensor of shape [b, n, dim_cond]

        Returns:
            All elements above extended with registers / zeros.
        """
        if self.num_registers == 0:
            return seqs, pair, mask, cond_seq  # Do nothing

        b, n, _ = seqs.shape
        dim_pair = pair.shape[-1]
        r = self.num_registers
        dim_cond = cond_seq.shape[-1]

        # Concatenate registers to sequence
        reg_expanded = self.registers[None, :, :]  # [1, r, dim_token]
        reg_expanded = reg_expanded.expand(b, -1, -1)  # [b, r, dim_token]
        seqs = torch.cat([reg_expanded, seqs], dim=1)  # [b, r+n, dim_token]

        # Extend mask
        true_tensor = torch.ones(b, r, dtype=torch.bool, device=seqs.device)  # [b, r]
        mask = torch.cat([true_tensor, mask], dim=1)  # [b, r+n]

        # Extend pair representation with zeros; pair has shape [b, n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        # [b, n, n, pair_dim] -> [b, r+n, n, pair_dim]
        zero_pad_top = torch.zeros(
            b, r, n, dim_pair, device=seqs.device
        )  # [b, r, n, dim_pair]
        pair = torch.cat([zero_pad_top, pair], dim=1)  # [b, r+n, n, dim_pair]
        # [b, r+n, n, pair_dim] -> [b, r+n, r+n, pair_dim]
        zero_pad_left = torch.zeros(
            b, r + n, r, dim_pair, device=seqs.device
        )  # [b, r+n, r, dim_pair]
        pair = torch.cat([zero_pad_left, pair], dim=2)  # [b, r+n, r+n, dim+pair]

        # Extend cond
        zero_tensor = torch.zeros(
            b, r, dim_cond, device=seqs.device
        )  # [b, r, dim_cond]
        cond_seq = torch.cat([zero_tensor, cond_seq], dim=1)  # [b, r+n, dim_cond]

        return seqs, pair, mask, cond_seq

    def _undo_registers(self, seqs, pair, mask):
        """
        Undoes register padding.

        Args:
            - seqs: sequence representation, shape [b, r+n, dim_token]
            - pair: pair representation, shape [b, r+n, r+n, dim_pair]
            - mask: binary mask, shape [b, r+n]

        Returns:
            All three elements with the register padding removed.
        """
        if self.num_registers == 0:
            return seqs, pair, mask
        r = self.num_registers
        return seqs[:, r:, :], pair[:, r:, r:, :], mask[:, r:]

    def forward(self, batch_nn: Dict[str, torch.Tensor], force_moe_capacity: bool = True) -> Dict[str, torch.Tensor]:
        """
        Runs the network.

        Args:
            batch_nn: dictionary with keys
                - "x_t": tensor of shape [b, n, 3]
                - "t": tensor of shape [b]
                - "mask": binary tensor of shape [b, n]
                - "x_sc" (optional): tensor of shape [b, n, 3]
                - "cath_code" (optional): list of cath codes [b, ?]
                - And potentially others... All in the data batch.

        Returns:
            Predicted clean coordinates, shape [b, n, 3].
        """
        mask = batch_nn["mask"]

        # Conditioning variables
        c = self.cond_factory(batch_nn)  # [b, n, dim_cond]
        c = self.transition_c_2(self.transition_c_1(c, mask), mask)  # [b, n, dim_cond]

        # Prepare input - coordinates and initial sequence representation from features
        coors_3d = batch_nn["x_t"] * mask[..., None]  # [b, n, 3]
        coors_embed = (
            self.linear_3d_embed(coors_3d) * mask[..., None]
        )  # [b, n, token_dim]
        seq_f_repr = self.init_repr_factory(batch_nn)  # [b, n, token_dim]
        seqs = coors_embed + seq_f_repr  # [b, n, token_dim]
        seqs = seqs * mask[..., None]  # [b, n, token_dim]

        # Pair representation
        pair_rep = self.pair_repr_builder(batch_nn)  # [b, n, n, pair_dim]

        # Apply registers
        seqs, pair_rep, mask, c = self._extend_w_registers(seqs, pair_rep, mask, c)

        # Run trunk
        for i in range(self.nlayers):
            seqs = self.transformer_layers[i](
                seqs, pair_rep, c, mask, force_moe_capacity
            )  # [b, n, token_dim]

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)

        # Get final coordinates
        final_coors = self.coors_3d_decoder(seqs) * mask[..., None]  # [b, n, 3]
        nn_out = {"coors_pred": final_coors}
        return nn_out

