from typing import Dict

import torch
import torch.nn as nn

from src.model.components.feature_factory import FeatureFactory
from src.model.components.pair_bias_attn import PairBiasAttention
from src.model.components.af3_modules import (
    AdaptiveLayerNorm,
    AdaptiveLayerNormOutputScale,
    Transition,
)
from src.model.components.protein_transformer import PairReprBuilder, MultiheadAttnAndTransition


class AtomTransformer(nn.Module):
    def __init__(self, **kwargs):
        super(AtomTransformer, self).__init__()

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
            feats=kwargs["feats_init_aa"],
            dim_feats_out=kwargs["token_dim"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        # To get conditioning variables
        self.cond_factory = FeatureFactory(
            feats=kwargs["feats_cond_aa"],
            dim_feats_out=kwargs["dim_cond"],
            use_ln_out=False,
            mode="seq",
            **kwargs,
        )

        self.transition_c_1 = Transition(kwargs["dim_cond"], expansion_factor=2)
        self.transition_c_2 = Transition(kwargs["dim_cond"], expansion_factor=2)

        # To get pair representation
        self.pair_repr_builder = PairReprBuilder(
            feats_repr=kwargs["feats_pair_repr_aa"],
            feats_cond=kwargs["feats_pair_cond_aa"],
            dim_feats_out=kwargs["pair_repr_dim"],
            dim_cond_pair=kwargs["dim_cond"],
            **kwargs,
        )

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
                )
                for _ in range(self.nlayers)
            ]
        )

        self.coors_3d_decoder = torch.nn.Sequential(
            torch.nn.LayerNorm(kwargs["token_dim"]),
            torch.nn.Linear(kwargs["token_dim"], 42, bias=False),  # 14 atoms per residue
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

    def forward(self, x_t, t, r, batch_nn: Dict[str, torch.Tensor]):
        mask = batch_nn["mask"]
        atom_mask = batch_nn["atom_mask"]  # [b, n * 14]
        b, nres = mask.shape

        batch_nn.update({
            "t": t,
            "r": r,
            "x_t": x_t,
        })

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
                seqs, pair_rep, c, mask
            )  # [b, n, token_dim]

        # Undo registers
        seqs, pair_rep, mask = self._undo_registers(seqs, pair_rep, mask)

        # Get final coordinates
        final_coors = self.coors_3d_decoder(seqs).view(b, nres * 14, 3) * atom_mask[..., None]
        return final_coors
