# network for generative modeling of residue-based particles
from functools import partial
from typing import Any, Optional, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.model.components.primitives import LinearNoBias, LayerNorm, Transition
from src.model.components.transformer import AtomAttentionEncoder, DiffusionTransformer


class ResGenNet(nn.Module):
    def __init__(
        self,
        s_input: int = 1280,
        s_trunk: int = 384,
        z_trunk: int = 128,
        s_atom: int = 128,
        z_atom: int = 16,
        s_noise: int = 256,
        z_template: int = 128,
        n_layers: int = 8,
        n_attn_heads: int = 8,
        sigma_data: float = 16.0
    ):
        super(ResGenNet, self).__init__()

        ### required components
        # embedding module, take sequence and reference features as input
        self.seq_embedder = SeqEmbedder(
            s_input=s_input,
            s_trunk=s_trunk,
            z_trunk=z_trunk,
            s_atom=s_atom,
            z_atom=z_atom,
            s_noise=s_noise,
        )

        # template module, take structure template as input
        self.template_embedder = TemplateEmbedder(
            z_template=z_template,
        )

        # main chunk
        self.linear_no_bias_a = LinearNoBias(in_features=3, out_features=s_trunk)
        self.layer_norm_a1 = LayerNorm(s_trunk)
        self.linear_no_bias_a1 = LinearNoBias(in_features=s_trunk, out_features=s_trunk)
        self.diffusion_transformer = DiffusionTransformer(
            c_a=s_trunk,
            c_s=s_trunk,
            c_z=z_trunk,
            n_blocks=n_layers,
            n_heads=n_attn_heads,
        )

        for block in self.diffusion_transformer.blocks:
            block.attention_pair_bias.layernorm_a.zero_init()
            block.conditioned_transition_block.adaln.zero_init()

        # output module
        self.layer_norm_o = LayerNorm(s_trunk)
        self.linear_no_bias_o = LinearNoBias(in_features=s_trunk, out_features=3)

        nn.init.zeros_(self.linear_no_bias_o.weight)

        self.sigma_data = sigma_data

    def forward(
        self,
        noisy_structure: torch.Tensor,
        plm_embedding: torch.Tensor,
        feature_dict: Dict[str, torch.Tensor],
        noise_scale: torch.Tensor,
        template: Optional[torch.Tensor],
    ):
        # scaling noisy structure
        a_noisy = noisy_structure / torch.sqrt(self.sigma_data ** 2 + noise_scale ** 2)[..., None, None]

        # network part
        s_embed, z_embed = self.seq_embedder(plm_embedding, feature_dict, noise_scale)

        a_embed = self.linear_no_bias_a(a_noisy)
        a_embed += self.linear_no_bias_a1(self.layer_norm_a1(s_embed))

        if template is not None:
            z_embed += self.template_embedder(template)

        a_out = self.diffusion_transformer(
            a=a_embed,
            s=s_embed,
            z=z_embed,
        )
        a_out = self.linear_no_bias_o(self.layer_norm_o(a_out))

        # reverse-scaling for output structure
        s_ratio = (noise_scale / self.sigma_data)[..., None, None].to(a_out.dtype)
        a_out = (
            1 / (1 + s_ratio ** 2) * noisy_structure
            + noise_scale[..., None, None] / torch.sqrt(1 + s_ratio ** 2) * a_out
        ).to(a_out.dtype)

        return a_out


class SeqEmbedder(nn.Module):
    def __init__(
        self,
        s_input: int = 1280,
        s_trunk: int = 384,
        z_trunk: int = 128,
        s_atom: int = 128,
        z_atom: int = 16,
        s_noise: int = 256,
        sigma_data: float = 16.0,
    ):
        super(SeqEmbedder, self).__init__()
        self.s_input = s_input
        self.s_trunk = s_trunk
        self.z_trunk = z_trunk
        self.s_atom = s_atom
        self.z_atom = z_atom
        self.s_noise = s_noise
        self.sigma_data = sigma_data

        # initial atom embedding
        # self.initial_atom_enbedding = AtomAttentionEncoder(
        #     c_atom=self.s_atom,
        #     c_atompair=self.z_atom,
        #     c_token=self.s_trunk,
        #     has_coords=False,
        # )

        # language model embedding (single embedding)
        self.layernorm_s = LayerNorm(s_input)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.s_input,
            out_features=self.s_trunk,
        )

        # noise scale fourier embedding
        self.fourier_embedding = FourierEmbedding(c=self.s_noise)
        self.layernorm_n = LayerNorm(self.s_noise)
        self.linear_no_bias_n = LinearNoBias(
            in_features=self.s_noise, out_features=self.s_trunk
        )

        self.transition_s1 = Transition(c_in=self.s_trunk, n=2)
        self.transition_s2 = Transition(c_in=self.s_trunk, n=2)

        # relative positional encoding
        self.relative_position_encoding = RelativePositionEncoding(c_z=self.z_trunk)

        # pair embedding
        self.layernorm_z = LayerNorm(self.z_trunk)
        self.linear_no_bias_z = LinearNoBias(in_features=self.z_trunk, out_features=self.z_trunk)

        self.transition_z1 = Transition(c_in=self.z_trunk, n=2)
        self.transition_z2 = Transition(c_in=self.z_trunk, n=2)

    def forward(
        self,
        plm_embedding: torch.Tensor,
        feature_dict: Dict[str, torch.Tensor],
        noise_scale: float,
    ):
        B, L, _ = plm_embedding.shape

        # process noise scale
        noise_n = self.fourier_embedding(t_hat_noise_level=torch.log(input=noise_scale / self.sigma_data) / 4).to(plm_embedding.dtype)
        noise_n = self.linear_no_bias_n(self.layernorm_n(noise_n)).unsqueeze(-2)

        s_init = self.linear_no_bias_s(self.layernorm_s(plm_embedding)).unsqueeze(-3)
        s_init = s_init + noise_n

        s_trunk = self.transition_s1(s_init)
        s_trunk = self.transition_s2(s_trunk)

        z_init = self.relative_position_encoding(feature_dict)
        z_init = self.linear_no_bias_z(self.layernorm_z(z_init))
        z_trunk = self.transition_z1(z_init)
        z_trunk = self.transition_z2(z_trunk)

        return s_trunk, z_trunk.unsqueeze(-4).expand(-1, s_trunk.shape[1], -1, -1, -1)


class TemplateEmbedder(nn.Module):
    def __init__(
        self,
        z_template: int = 128
    ):
        super(TemplateEmbedder, self).__init__()
        self.z_template = z_template

        self.linear_no_bias_t1 = LinearNoBias(in_features=3, out_features=self.z_template)
        self.linear_no_bias_t2 = LinearNoBias(in_features=1, out_features=self.z_template)

        self.transition_t1 = Transition(c_in=self.z_template, n=2)
        self.transition_t2 = Transition(c_in=self.z_template, n=2)

    def forward(
        self,
        template_coord: torch.Tensor,
    ):
        template_dist = template_coord[..., None, :] - template_coord[..., None, :, :]

        z_temp = self.linear_no_bias_t1(template_dist)
        z_temp += self.linear_no_bias_t2(1 / (1 + (z_temp ** 2).sum(dim=-1, keepdim=True)))

        z_temp = self.transition_t1(z_temp)
        z_temp = self.transition_t2(z_temp)

        return z_temp


class RelativePositionEncoding(nn.Module):
    def __init__(self, r_max: int = 32, s_max: int = 2, c_z: int = 128) -> None:
        """
        Args:
            r_max (int, optional): Relative position indices clip value. Defaults to 32.
            s_max (int, optional): Relative chain indices clip value. Defaults to 2.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
        """
        super(RelativePositionEncoding, self).__init__()
        self.r_max = r_max
        self.s_max = s_max
        self.c_z = c_z
        self.linear_no_bias = LinearNoBias(
            in_features=(4 * self.r_max + 4), out_features=self.c_z
        )
        self.input_feature = {
            "asym_id": 1,
            "residue_index": 1,
            "entity_id": 1,
            "sym_id": 1,
            "token_index": 1,
        }

    def forward(self, input_feature_dict: dict[str, Any]) -> torch.Tensor:
        """
        Args:
            input_feature_dict (Dict[str, Any]): input meta feature dict.
            asym_id / residue_index / entity_id / sym_id / token_index
                [..., N_tokens]
        Returns:
            torch.Tensor: relative position encoding
                [..., N_token, N_token, c_z]
        """
        b_same_chain = (
                input_feature_dict["chain_index"][..., :, None]
                == input_feature_dict["chain_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
                input_feature_dict["residue_index"][..., :, None]
                == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]

        d_residue = torch.clip(
            input=input_feature_dict["residue_index"][..., :, None]
                  - input_feature_dict["residue_index"][..., None, :]
                  + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain + (1 - b_same_chain) * (
                            2 * self.r_max + 1
                    )  # [..., N_token, N_token]
        a_rel_pos = F.one_hot(d_residue, 2 * (self.r_max + 1))

        d_token = torch.clip(
            input=input_feature_dict["token_index"][..., :, None]
                  - input_feature_dict["token_index"][..., None, :]
                  + self.r_max,
            min=0,
            max=2 * self.r_max,
        ) * b_same_chain * b_same_residue + (1 - b_same_chain * b_same_residue) * (
                          2 * self.r_max + 1
                  )  # [..., N_token, N_token]
        a_rel_token = F.one_hot(d_token, 2 * (self.r_max + 1))

        p = self.linear_no_bias(
            torch.cat(
                [a_rel_pos, a_rel_token],
                dim=-1,
            ).float()
        )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
        return p


class FourierEmbedding(nn.Module):
    """
    Implements Algorithm 22 in AF3
    """

    def __init__(self, c: int, seed: int = 42) -> None:
        """
        Args:
            c (int): embedding dim.
        """
        super(FourierEmbedding, self).__init__()
        self.c = c
        self.seed = seed
        generator = torch.Generator()
        generator.manual_seed(seed)
        w_value = torch.randn(size=(c,), generator=generator)
        self.w = nn.Parameter(w_value, requires_grad=False)
        b_value = torch.randn(size=(c,), generator=generator)
        self.b = nn.Parameter(b_value, requires_grad=False)

    def forward(self, t_hat_noise_level: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]

        Returns:
            torch.Tensor: the output fourier embedding
                [..., N_sample, c]
        """
        return torch.cos(
            input=2 * torch.pi * (t_hat_noise_level.unsqueeze(dim=-1) * self.w + self.b)
        )

