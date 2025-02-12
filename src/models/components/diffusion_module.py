# Copyright 2024 ByteDance and/or its affiliates.
#
# Licensed under the Attribution-NonCommercial 4.0 International
# License (the "License"); you may not use this file except in
# compliance with the License. You may obtain a copy of the
# License at
import math
from functools import partial
from random import random
#     https://creativecommons.org/licenses/by-nc/4.0/

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Optional, Union

import torch
import torch.nn as nn

from src.models.components.embedders import FourierEmbedding, RelativePositionEncoding
from src.models.components.primitives import LinearNoBias, Transition, LayerNorm
# from src.models.components.transformer import (
#     AtomAttentionDecoder,
#     AtomAttentionEncoder,
#     DiffusionTransformer,
# )
from src.utils.model_utils import expand_at_dim, get_checkpoint_fn
from src.common.geo_utils import calc_distogram


class DiffusionConditioning(nn.Module):
    """
    Implements Algorithm 21 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_z: int = 128,
        c_s: int = 384,
        c_s_inputs: int = 449,
        c_noise_embedding: int = 256,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_s_inputs (int, optional): input embedding dim from InputEmbedder. Defaults to 449.
            c_noise_embedding (int, optional): noise embedding dim. Defaults to 256.
        """
        super(DiffusionConditioning, self).__init__()
        self.sigma_data = sigma_data
        self.c_z = c_z
        self.c_s = c_s
        self.c_s_inputs = c_s_inputs
        # Line1-Line3:
        self.relpe = RelativePositionEncoding(c_z=c_z)
        self.layernorm_z = LayerNorm(2 * self.c_z)
        self.linear_no_bias_z = LinearNoBias(
            in_features=2 * self.c_z, out_features=self.c_z
        )
        # Line3-Line5:
        self.transition_z1 = Transition(c_in=self.c_z, n=2)
        self.transition_z2 = Transition(c_in=self.c_z, n=2)

        # Line6-Line7
        self.layernorm_s = LayerNorm(self.c_s + self.c_s_inputs)
        self.linear_no_bias_s = LinearNoBias(
            in_features=self.c_s + self.c_s_inputs, out_features=self.c_s
        )
        # Line8-Line9
        self.fourier_embedding = FourierEmbedding(c=c_noise_embedding)
        self.layernorm_n = LayerNorm(c_noise_embedding)
        self.linear_no_bias_n = LinearNoBias(
            in_features=c_noise_embedding, out_features=self.c_s
        )
        # Line10-Line12
        self.transition_s1 = Transition(c_in=self.c_s, n=2)
        self.transition_s2 = Transition(c_in=self.c_s, n=2)
        print(f"Diffusion Module has {self.sigma_data}")

    def forward(
        self,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        s_trunk: torch.Tensor,
        z_trunk: torch.Tensor,
        inplace_safe: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            t_hat_noise_level (torch.Tensor): the noise level
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations.
        Returns:
            tuple[torch.Tensor, torch.Tensor]: embeddings s and z
                - s (torch.Tensor): [..., N_sample, N_tokens, c_s]
                - z (torch.Tensor): [..., N_tokens, N_tokens, c_z]
        """
        # Pair conditioning
        pair_z = torch.cat(
            tensors=[z_trunk, self.relpe(input_feature_dict)], dim=-1
        )  # [..., N_tokens, N_tokens, 2*c_z]
        pair_z = self.linear_no_bias_z(self.layernorm_z(pair_z))
        if inplace_safe:
            pair_z += self.transition_z1(pair_z)
            pair_z += self.transition_z2(pair_z)
        else:
            pair_z = pair_z + self.transition_z1(pair_z)
            pair_z = pair_z + self.transition_z2(pair_z)
        # Single conditioning
        single_s = torch.cat(
            tensors=[s_trunk, s_inputs], dim=-1
        )  # [..., N_tokens, c_s + c_s_inputs]
        single_s = self.linear_no_bias_s(self.layernorm_s(single_s))
        noise_n = self.fourier_embedding(
            t_hat_noise_level=torch.log(input=t_hat_noise_level / self.sigma_data) / 4
        ).to(
            single_s.dtype
        )  # [..., N_sample, c_in]
        single_s = single_s.unsqueeze(dim=-3) + self.linear_no_bias_n(
            self.layernorm_n(noise_n)
        ).unsqueeze(
            dim=-2
        )  # [..., N_sample, N_tokens, c_s]
        if inplace_safe:
            single_s += self.transition_s1(single_s)
            single_s += self.transition_s2(single_s)
        else:
            single_s = single_s + self.transition_s1(single_s)
            single_s = single_s + self.transition_s2(single_s)
        if not self.training and pair_z.shape[-2] > 2000:
            torch.cuda.empty_cache()
        return single_s, pair_z


def get_positional_embedding(indices, embedding_dim, max_len=2056):
    """Creates sine / cosine positional embeddings from a prespecified indices.

    Args:
        indices: offsets of size [..., N_edges] of type integer
        max_len: maximum length.
        embedding_dim: dimension of the embeddings to create

    Returns:
        positional embedding of shape [N, embedding_dim]
    """
    K = torch.arange(embedding_dim//2, device=indices.device)
    pos_embedding_sin = torch.sin(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding_cos = torch.cos(
        indices[..., None] * math.pi / (max_len**(2*K[None]/embedding_dim))).to(indices.device)
    pos_embedding = torch.cat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding


class EmbeddingModule(nn.Module):
    def __init__(self,
                 init_embed_size: int = 256,
                 node_embed_size: int = 256,
                 edge_embed_size: int = 128,
                 embedding_size: int = 1280,
                 sigma_data: float = 16.0,
                 num_bins: int = 22,
                 min_bin: float = 1e-5,
                 max_bin: float = 20.0,
                 self_conditioning: bool = True,
                 ):
        super(EmbeddingModule, self).__init__()
        pos_embed_size = init_embed_size
        t_embed_size = init_embed_size

        # time embedding
        node_in_dim = t_embed_size + 1
        edge_in_dim = (t_embed_size + 1) * 2

        # positional embedding
        node_in_dim += pos_embed_size
        edge_in_dim += pos_embed_size

        self.sigma_data = sigma_data

        self.fourier_embedding = FourierEmbedding(256)
        self.plm_embed = nn.Sequential(
            nn.Linear(embedding_size + init_embed_size, init_embed_size),
            LayerNorm(init_embed_size),
        )

        self.node_embed = nn.Sequential(
            nn.Linear(node_in_dim, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            nn.ReLU(),
            nn.Linear(node_embed_size, node_embed_size),
            LayerNorm(node_embed_size),
        )

        # self-conditioning trick used in RFDiffusion
        self.self_conditioning = self_conditioning
        if self_conditioning:
            edge_in_dim += num_bins

        self.edge_embed = nn.Sequential(
            nn.Linear(edge_in_dim, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            nn.ReLU(),
            nn.Linear(edge_embed_size, edge_embed_size),
            LayerNorm(edge_embed_size),
        )

        self.position_embed = partial(
            get_positional_embedding, embedding_dim=pos_embed_size
        )
        self.distogram_embed = partial(
            calc_distogram,
            min_bin=min_bin,
            max_bin=max_bin,
            num_bins=num_bins,
        )

    def forward(
            self,
            residue_idx,
            t_hat,
            fixed_mask,
            plm_embedding,
            self_conditioning_ca,
    ):
        """
        Args:
            residue_idx: [..., N] Positional sequence index for each residue.
            t: Sampled t in [0, 1].
            fixed_mask: mask of fixed (motif) residues.
            self_conditioning_ca: [..., N, 3] Ca positions of self-conditioning
                input.

        Returns:
            node_embed: [B, N, D_node]
            edge_embed: [B, N, N, D_edge]
        """
        B, N_sample, L = plm_embedding.shape[:3]
        fixed_mask = fixed_mask[..., None].float()
        node_feats = []
        pair_feats = []

        # configure time embedding
        t_embed = torch.tile(self.fourier_embedding(0.25 * torch.log(t_hat / self.sigma_data))[:, :, None, :], (1, 1, L, 1))
        t_embed = self.plm_embed(
            torch.cat([t_embed, plm_embedding], dim=-1)
        )
        t_embed = torch.cat([t_embed, fixed_mask], dim=-1)
        node_feats.append(t_embed)

        # make pair embedding from 1d time feats
        concat_1d = torch.cat(
            [torch.tile(t_embed[:, :, :, None, :], (1, 1, 1, L, 1)),
             torch.tile(t_embed[:, :, None, :, :], (1, 1, L, 1, 1))],
            dim=-1).float().reshape([B, N_sample, L ** 2, -1])
        pair_feats.append(concat_1d)

        # positional embedding
        node_feats.append(self.position_embed(residue_idx).expand([B, N_sample, L, -1]))

        # relative 2d positional embedding
        rel_seq_offset = residue_idx[:, :, :, None] - residue_idx[:, :, None, :]
        rel_seq_offset = rel_seq_offset.reshape([B, 1, L ** 2])
        pair_feats.append(self.position_embed(rel_seq_offset).expand([B, N_sample, L ** 2, -1]))

        # self-conditioning distogram of C-alpha atoms
        if self.self_conditioning:
            self_conditioning_ca = self_conditioning_ca.unsqueeze(1).expand([B, N_sample, L, 3])
            ca_dist = self.distogram_embed(self_conditioning_ca)
            pair_feats.append(ca_dist.reshape([B, N_sample, L ** 2, -1]))

        node_embed = self.node_embed(torch.cat(node_feats, dim=-1).float())
        edge_embed = self.edge_embed(torch.cat(pair_feats, dim=-1).float())
        edge_embed = edge_embed.reshape([B, N_sample, L, L, -1])
        return node_embed, edge_embed


class DiffusionSchedule:
    def __init__(
        self,
        sigma_data: float = 16.0,
        s_max: float = 160.0,
        s_min: float = 4e-4,
        p: float = 7.0,
        dt: float = 1 / 200,
        p_mean: float = -1.2,
        p_std: float = 1.5,
    ) -> None:
        """
        Args:
            sigma_data (float, optional): The standard deviation of the data. Defaults to 16.0.
            s_max (float, optional): The maximum noise level. Defaults to 160.0.
            s_min (float, optional): The minimum noise level. Defaults to 4e-4.
            p (float, optional): The exponent for the noise schedule. Defaults to 7.0.
            dt (float, optional): The time step size. Defaults to 1/200.
            p_mean (float, optional): The mean of the log-normal distribution for noise level sampling. Defaults to -1.2.
            p_std (float, optional): The standard deviation of the log-normal distribution for noise level sampling. Defaults to 1.5.
        """
        self.sigma_data = sigma_data
        self.s_max = s_max
        self.s_min = s_min
        self.p = p
        self.dt = dt
        self.p_mean = p_mean
        self.p_std = p_std
        # self.T
        self.T = int(1 / dt) + 1  # 201

    def get_train_noise_schedule(self) -> torch.Tensor:
        return self.sigma_data * torch.exp(self.p_mean + self.p_std * torch.randn(1))

    def get_inference_noise_schedule(self) -> torch.Tensor:
        time_step_lists = torch.arange(start=0, end=1 + 1e-10, step=self.dt)
        inference_noise_schedule = (
            self.sigma_data
            * (
                self.s_max ** (1 / self.p)
                + time_step_lists
                * (self.s_min ** (1 / self.p) - self.s_max ** (1 / self.p))
            )
            ** self.p
        )
        return inference_noise_schedule


class DiffusionModule(nn.Module):
    """
    Implements Algorithm 20 in AF3
    """

    def __init__(
        self,
        sigma_data: float = 16.0,
        c_atom: int = 128,
        c_atompair: int = 16,
        c_token: int = 768,
        c_s: int = 384,
        c_z: int = 128,
        c_s_inputs: int = 1280,
        embedding_module: nn.Module=None,
        atom_encoder: nn.Module=None,
        transformer: nn.Module=None,
        atom_decoder: nn.Module=None,
        blocks_per_ckpt: Optional[int] = None,
        use_fine_grained_checkpoint: bool = False,
        initialization: Optional[dict[str, Union[str, float, bool]]] = None,
        training: bool = True,
    ) -> None:
        """
        Args:
            sigma_data (torch.float, optional): the standard deviation of the data. Defaults to 16.0.
            c_atom (int, optional): embedding dim for atom feature. Defaults to 128.
            c_atompair (int, optional): embedding dim for atompair feature. Defaults to 16.
            c_token (int, optional): feature channel of token (single a). Defaults to 768.
            c_s (int, optional):  hidden dim [for single embedding]. Defaults to 384.
            c_z (int, optional): hidden dim [for pair embedding]. Defaults to 128.
            c_s_inputs (int, optional): hidden dim [for single input embedding]. Defaults to 449.
            atom_encoder (dict[str, int], optional): configs in AtomAttentionEncoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            transformer (dict[str, int], optional): configs in DiffusionTransformer. Defaults to {"n_blocks": 24, "n_heads": 16}.
            atom_decoder (dict[str, int], optional): configs in AtomAttentionDecoder. Defaults to {"n_blocks": 3, "n_heads": 4}.
            blocks_per_ckpt: number of atom_encoder/transformer/atom_decoder blocks in each activation checkpoint
                Size of each chunk. A higher value corresponds to fewer
                checkpoints, and trades memory for speed. If None, no checkpointing is performed.
            use_fine_grained_checkpoint: whether use fine-gained checkpoint for finetuning stage 2
                only effective if blocks_per_ckpt is not None.
            initialization: initialize the diffusion module according to initialization config.
        """

        super(DiffusionModule, self).__init__()
        self.sigma_data = sigma_data
        self.c_atom = c_atom
        self.c_atompair = c_atompair
        self.c_token = c_token
        self.c_s_inputs = c_s_inputs
        self.c_s = c_s
        self.c_z = c_z

        # Grad checkpoint setting
        self.blocks_per_ckpt = blocks_per_ckpt
        self.use_fine_grained_checkpoint = use_fine_grained_checkpoint

        # self.diffusion_conditioning = DiffusionConditioning(
        #     sigma_data=self.sigma_data, c_z=c_z, c_s=c_s, c_s_inputs=c_s_inputs
        # )
        self.diffusion_conditioning = embedding_module
        self.atom_attention_encoder = atom_encoder

        # Alg20: line4
        self.layernorm_s = LayerNorm(c_s)
        self.linear_no_bias_s = LinearNoBias(in_features=c_s, out_features=c_token)
        self.diffusion_transformer = transformer
        self.layernorm_a = LayerNorm(c_token)
        self.atom_attention_decoder = atom_decoder

        self.init_parameters(initialization)
        self.training = training

    def init_parameters(self, initialization: dict):
        """
        Initializes the parameters of the diffusion module according to the provided initialization configuration.

        Args:
            initialization (dict): A dictionary containing initialization settings.
        """
        if initialization.get("zero_init_condition_transition", False):
            self.diffusion_conditioning.transition_z1.zero_init()
            self.diffusion_conditioning.transition_z2.zero_init()
            self.diffusion_conditioning.transition_s1.zero_init()
            self.diffusion_conditioning.transition_s2.zero_init()

        self.atom_attention_encoder.linear_init(
            zero_init_atom_encoder_residual_linear=initialization.get(
                "zero_init_atom_encoder_residual_linear", False
            ),
            he_normal_init_atom_encoder_small_mlp=initialization.get(
                "he_normal_init_atom_encoder_small_mlp", False
            ),
            he_normal_init_atom_encoder_output=initialization.get(
                "he_normal_init_atom_encoder_output", False
            ),
        )

        if initialization.get("glorot_init_self_attention", False):
            for (
                block
            ) in (
                self.atom_attention_encoder.atom_transformer.diffusion_transformer.blocks
            ):
                block.attention_pair_bias.glorot_init()

        for block in self.diffusion_transformer.blocks:
            if initialization.get("zero_init_adaln", False):
                block.attention_pair_bias.layernorm_a.zero_init()
                block.conditioned_transition_block.adaln.zero_init()
            if initialization.get("zero_init_residual_condition_transition", False):
                nn.init.zeros_(
                    block.conditioned_transition_block.linear_nobias_b.weight
                )

        if initialization.get("zero_init_atom_decoder_linear", False):
            nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_a.weight)

        if initialization.get("zero_init_dit_output", False):
            nn.init.zeros_(self.atom_attention_decoder.linear_no_bias_out.weight)

    def f_forward(
        self,
        r_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """The raw network to be trained.
        As in EDM equation (7), this is F_theta(c_in * x, c_noise(sigma)).
        Here, c_noise(sigma) is computed in Conditioning module.

        Args:
            r_noisy (torch.Tensor): scaled x_noisy (i.e., c_in * x)
                [..., N_sample, N_atom, 3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input feature
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: coordinates update
                [..., N_sample, N_atom, 3]
        """
        N_sample = r_noisy.size(-3)
        assert t_hat_noise_level.size(-1) == N_sample

        blocks_per_ckpt = self.blocks_per_ckpt
        if not torch.is_grad_enabled():
            blocks_per_ckpt = None
        # Conditioning, shared across difference samples
        # Diffusion_conditioning consumes 7-8G when token num is 768,
        # use checkpoint here if blocks_per_ckpt is not None.
        if blocks_per_ckpt:
            checkpoint_fn = get_checkpoint_fn()
            s_single, z_pair = checkpoint_fn(
                self.diffusion_conditioning,
                input_feature_dict["residue_index"].unsqueeze(1),
                t_hat_noise_level,
                input_feature_dict["seq_mask"].unsqueeze(1).expand(-1, N_sample, -1),
                s_inputs,
                input_feature_dict["ref_com"],
            )
        else:
            try:
                s_single, z_pair = self.diffusion_conditioning(
                    input_feature_dict["residue_index"].unsqueeze(1),
                    t_hat_noise_level,
                    input_feature_dict["seq_mask"].unsqueeze(1).expand(-1, N_sample, -1),
                    s_inputs,
                    input_feature_dict["ref_com"],
                )
            except:
                print(input_feature_dict["accession_code"])
            # [..., N_sample, N_token, c_s], [..., N_token, N_token, c_z]

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            a_token, q_skip, c_skip, p_skip = checkpoint_fn(
                self.atom_attention_encoder,
                input_feature_dict,
                r_noisy,
                s_single,
                z_pair,
                inplace_safe,
                chunk_size,
            )
        else:
            # Sequence-local Atom Attention and aggregation to coarse-grained tokens
            a_token, q_skip, c_skip, p_skip = self.atom_attention_encoder(
                input_feature_dict=input_feature_dict,
                r_l=r_noisy,
                s=s_single,
                z=z_pair,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )
        # Full self-attention on token level.
        if inplace_safe:
            a_token += self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        else:
            a_token = a_token + self.linear_no_bias_s(
                self.layernorm_s(s_single)
            )  # [..., N_sample, N_token, c_token]
        a_token = self.diffusion_transformer(
            a=a_token,
            s=s_single,
            z=z_pair,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        a_token = self.layernorm_a(a_token)

        # Fine-grained checkpoint for finetuning stage 2 (token num: 768) for avoiding OOM
        if blocks_per_ckpt and self.use_fine_grained_checkpoint:
            checkpoint_fn = get_checkpoint_fn()
            r_update = checkpoint_fn(
                self.atom_attention_decoder,
                input_feature_dict,
                a_token,
                q_skip,
                c_skip,
                p_skip,
                inplace_safe,
                chunk_size,
            )
        else:
            # Broadcast token activations to atoms and run Sequence-local Atom Attention
            r_update = self.atom_attention_decoder(
                input_feature_dict=input_feature_dict,
                a=a_token,
                q_skip=q_skip,
                c_skip=c_skip,
                p_skip=p_skip,
                inplace_safe=inplace_safe,
                chunk_size=chunk_size,
            )

        return r_update

    def forward(
        self,
        x_noisy: torch.Tensor,
        t_hat_noise_level: torch.Tensor,
        input_feature_dict: dict[str, Union[torch.Tensor, int, float, dict]],
        s_inputs: torch.Tensor,
        inplace_safe: bool = False,
        chunk_size: Optional[int] = None,
    ) -> torch.Tensor:
        """One step denoise: x_noisy, noise_level -> x_denoised

        Args:
            x_noisy (torch.Tensor): the noisy version of the input atom coords
                [..., N_sample, N_atom,3]
            t_hat_noise_level (torch.Tensor): the noise level, as well as the time step t
                [..., N_sample]
            input_feature_dict (dict[str, Union[torch.Tensor, int, float, dict]]): input meta feature dict
            s_inputs (torch.Tensor): single embedding from InputFeatureEmbedder
                [..., N_tokens, c_s_inputs]
            s_trunk (torch.Tensor): single feature embedding from PairFormer (Alg17)
                [..., N_tokens, c_s]
            z_trunk (torch.Tensor): pair feature embedding from PairFormer (Alg17)
                [..., N_tokens, N_tokens, c_z]
            inplace_safe (bool): Whether it is safe to use inplace operations. Defaults to False.
            chunk_size (Optional[int]): Chunk size for memory-efficient operations. Defaults to None.

        Returns:
            torch.Tensor: the denoised coordinates of x
                [..., N_sample, N_atom,3]
        """
        # Scale positions to dimensionless vectors with approximately unit variance
        # As in EDM:
        #     r_noisy = (c_in * x_noisy)
        #     where c_in = 1 / sqrt(sigma_data^2 + sigma^2)
        r_noisy = (
            x_noisy
            / torch.sqrt(self.sigma_data**2 + t_hat_noise_level**2)[..., None, None]
        )

        # Compute the update given r_noisy (the scaled x_noisy)
        # As in EDM:
        #     r_update = F(r_noisy, c_noise(sigma))
        r_update = self.f_forward(
            r_noisy=r_noisy,
            t_hat_noise_level=t_hat_noise_level,
            input_feature_dict=input_feature_dict,
            s_inputs=s_inputs,
            inplace_safe=inplace_safe,
            chunk_size=chunk_size,
        )

        # Rescale updates to positions and combine with input positions
        # As in EDM:
        #     D = c_skip * x_noisy + c_out * r_update
        #     c_skip = sigma_data^2 / (sigma_data^2 + sigma^2)
        #     c_out = (sigma_data * sigma) / sqrt(sigma_data^2 + sigma^2)
        #     s_ratio = sigma / sigma_data
        #     c_skip = 1 / (1 + s_ratio^2)
        #     c_out = sigma / sqrt(1 + s_ratio^2)

        s_ratio = (t_hat_noise_level / self.sigma_data)[..., None, None].to(
            r_update.dtype
        )
        x_denoised = (
            1 / (1 + s_ratio**2) * x_noisy
            + t_hat_noise_level[..., None, None] / torch.sqrt(1 + s_ratio**2) * r_update
        ).to(r_update.dtype)

        return x_denoised

