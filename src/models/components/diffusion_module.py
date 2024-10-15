#   Copyright (c) 2024 torchHelix Authors All Rights Reserved.
#
# Licensed under Creative Commons Attribution-NonCommercial-ShareAlike 4.0
# International License (the "License");  you may not use this file  except
# in compliance with the License. You may obtain a copy of the License at
#
#     http://creativecommons.org/licenses/by-nc-sa/4.0/
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Modules and utilities for the diffusion module."""

import copy
import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from scipy.spatial.transform import Rotation

from src.utils.model_utils import recompute_wrapper


def tile_batch_dim(batch, repeat_time):
    """
    Tile tensor along the batch dimension.
    Args:
        batch: a dict of torch tensor, or a pure torch tensor
        repeat_time: int
    """

    def _tile(x, repeat_time):
        if isinstance(x, torch.Tensor):
            shape = [repeat_time] + [1] * (len(x.shape) - 1)
            return torch.tile(x, shape)
        elif isinstance(x, list):
            return x * repeat_time
        else:
            raise ValueError(f'Unsupported type {type(x)}')

    if isinstance(batch, dict):
        new_batch = {}
        for name, x in batch.items():
            new_batch[name] = _tile(x, repeat_time)
        return new_batch
    else:
        return _tile(batch, repeat_time)


def insert_diff_batch_dim(batch, repeat_time, special_keys=None):
    """
    Insert a diff_batch_dim after the batch (0-th) dim.
    Args:
        batch: a dict of torch tensor, or a pure torch tensor
        repeat_time: int
        special_keys: list of keys whose repeat_time = 1
    """

    def _insert(x, repeat_time):
        if isinstance(x, torch.Tensor):
            return torch.repeat_interleave(x[:, None],
                                           repeat_time, dim=1)
        elif isinstance(x, list):
            return [[y] * repeat_time for y in x]
        else:
            raise ValueError(f'Unsupported type {type(x)}')

    if isinstance(batch, dict):
        new_batch = {}
        for name, x in batch.items():
            if not special_keys is None and name in special_keys:
                new_batch[name] = _insert(x, 1)
            else:
                new_batch[name] = _insert(x, repeat_time)
        return new_batch
    else:
        return _insert(batch, repeat_time)


def get_noise_schedule(sigma_data, s_max, s_min, p, step_size):
    """
    get_noise_schedule
    """
    t = torch.arange(0, 1 + step_size, step_size)
    t_tau = sigma_data * (s_max ** (1 / p) +
                          t * (s_min ** (1 / p) - s_max ** (1 / p))) ** p
    return t_tau


def CentreRandomAugmentation(x, mask, s_trans=1):
    """
    x: (B, N_atom, 3)
    mask: (B, N_atom)
    """
    mask = mask[..., None]
    mean_x = (x * mask).sum([1]) / (mask.sum([1]) + 1e-6)
    x = x - mean_x[:, None]
    B = x.shape[0]
    R = [Rotation.random().as_matrix() for _ in range(B)]
    R = torch.tensor(R, dtype=x.dtype).to(x.device)  # (B, 3, 3)
    t = s_trans * torch.normal(0., 1., size=[B, 1, 3]).to(x.device)
    x = x @ R + t
    x = x * mask
    return x


class DiffusionModule(nn.Module):
    """
    Diffusion Module
    """

    def __init__(self,
                 training: bool,
                 token_channel: int,
                 diffusion_token_channel: int,
                 diffusion_conditioning: nn.Module,
                 atom_attention_encoder: nn.Module,
                 diffusion_transformer_main: nn.Module,
                 atom_attention_decoder: nn.Module,
                 ):
        super(DiffusionModule, self).__init__()
        self.config = config
        self.global_config = global_config

        ## diffusion coefficient
        self.sigma_data = 16
        self.s_max = 160
        self.s_min = 4e-4
        self.p = 7
        self.gamma0 = 0.8
        self.gamma_min = 1.0
        self.lambda_ = 1.003
        self.eta = 1.5
        self.P_mean = -1.2
        self.P_std = 1.5

        token_channel = token_channel
        diffusion_token_channel = diffusion_token_channel

        ## network
        self.diffusion_conditioning = diffusion_conditioning
        self.atom_encoder = atom_attention_encoder

        self.ln1 = nn.LayerNorm(token_channel)
        self.lin1 = nn.Linear(token_channel, diffusion_token_channel, bias=False)
        self.diffusion_transformer = diffusion_transformer_main

        self.ln2 = nn.LayerNorm(diffusion_token_channel)
        self.atom_decoder = atom_attention_decoder
        
        self.training = training

    def _noise_schedule(self, step_num):
        assert step_num > 0
        step_size = 1.0 / step_num
        return get_noise_schedule(self.sigma_data,
                                  self.s_max, self.s_min, self.p, step_size)

    def _c_ckip(self, t_hat):
        return self.sigma_data ** 2 / (self.sigma_data ** 2 + t_hat ** 2)

    def _c_out(self, t_hat):
        return self.sigma_data * t_hat / torch.sqrt(self.sigma_data ** 2 + t_hat ** 2)

    def _forward_model(self, x_noisy, t_hat, batch, representations, return_r=False):
        """
        x_noisy: (B, N_atom, 3)
        t_hat: (B)
        """
        s_inputs = representations['single_inputs']  # (B, N_token, d1)
        s_trunk = representations['single']  # (B, N_token, d1)
        z_trunk = representations['pair']  # (raw_batch, N_token, N_token, d2)
        rel_pos_encoding = representations['rel_pos_encoding']  # (raw_batch, N_token, N_token, d2)
        # atom_mask = batch['all_atom_pos_mask']  # (B, N_atom)
        seq_mask = batch['seq_mask']  # (B, N_token)
        si, zij = self.diffusion_conditioning(
            t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, self.sigma_data)

        t_hat = t_hat.unsqueeze(1).unsqueeze(2)
        r_noisy = x_noisy / torch.sqrt(t_hat ** 2 + self.sigma_data ** 2)

        atom_token_uid = batch['ref_token2atom_idx']
        atom_mask = torch.ones_like(atom_token_uid)
        ai, ql_skip, cl_skip, p_lm_skip = self.atom_encoder(feature=batch, rl=r_noisy,
                                                            s_trunk=s_trunk, zij=zij)

        ai += self.lin1(self.ln1(si))
        beta = (1 - seq_mask[:, :, None] * seq_mask[:, None]) * (-1e8)
        ai = self.diffusion_transformer(ai, si, zij, beta=beta)

        ai = self.ln2(ai)
        r_update = self.atom_decoder(ai, ql_skip, cl_skip, p_lm_skip,
                                     atom_token_uid, atom_mask)

        x_out = self._c_ckip(t_hat) * x_noisy + self._c_out(t_hat) * r_update
        x_out = x_out * atom_mask[..., None]

        if return_r:
            return x_out, r_update * atom_mask[..., None]
        return x_out

    def forward(self, representations, batch):
        """forward"""
        x_noisy = batch['x_noisy']
        t_hat = batch['t_hat']
        return self._forward_model(x_noisy, t_hat, batch, representations)

    def sample_diffusion(self, representations, batch, step_num=200):
        """
        sample_diffusion
        """
        single_act = representations['single']  # (B, N, d1)
        atom_mask = batch['all_atom_pos_mask']
        B, N_atom = atom_mask.shape[:2]
        c_list = self._noise_schedule(step_num)
        x = c_list[0] * torch.normal(0., 1., size=[B, N_atom, 3])
        x = x.to(atom_mask.device)
        for i in range(1, len(c_list)):
            c_tau = c_list[i]
            c_tau_1 = c_list[i - 1]
            x = CentreRandomAugmentation(x, atom_mask)
            gamma = self.gamma0 if c_tau > self.gamma_min else 0
            t_hat = c_tau_1 * (gamma + 1)
            xi = self.lambda_ * torch.sqrt(t_hat ** 2 - c_tau_1 ** 2) * torch.normal(0., 1., size=[B, N_atom, 3])
            x_noisy = x + xi.to(x.device)
            x_denoised = self._forward_model(x_noisy,
                                             torch.tile(t_hat, [B]).to(x_noisy.device),
                                             batch, representations)
            delta = (x_noisy - x_denoised) / t_hat
            dt = c_tau - t_hat
            x = x_noisy + self.eta * dt * delta

        x *= atom_mask[..., None]
        ret = {
            'final_atom_positions': x,  # (B, N_atom, 3)
            'final_atom_mask': atom_mask,  # (B, N_atom)
        }
        return ret


class DiffusionConditioning(nn.Module):
    """
    DiffusionConditioning
    """

    def __init__(self,
                 token_channel: int,
                 token_pair_channel: int):
        super(DiffusionConditioning, self).__init__()

        token_channel = token_channel
        pair_channel = token_pair_channel

        self.pair_ln = nn.LayerNorm(pair_channel * 2)
        self.pair_lin = nn.Linear(pair_channel * 2, pair_channel, bias=False)
        self.pair_trans1 = Transition(pair_channel, n=2)
        self.pair_trans2 = Transition(pair_channel, n=2)

        self.single_ln1 = nn.LayerNorm(token_channel * 2 + 32 + 32 + 1)
        self.single_lin1 = nn.Linear(
            token_channel * 2 + 32 + 32 + 1, token_channel, bias=False)
        self.fourier_embedding = FourierEmbedding(256)
        self.single_ln2 = nn.LayerNorm(256)
        self.single_lin2 = nn.Linear(256, token_channel, bias=False)
        self.single_trans1 = Transition(token_channel, n=2)
        self.single_trans2 = Transition(token_channel, n=2)

    def forward(self, t_hat, rel_pos_encoding, s_inputs, s_trunk, z_trunk, sigma_data):
        """forward"""
        zij = torch.concat([z_trunk, rel_pos_encoding], -1)
        zij = self.pair_lin(self.pair_ln(zij))
        zij += self.pair_trans1(zij)
        zij += self.pair_trans2(zij)

        si = torch.concat([s_trunk, s_inputs], -1)
        si = self.single_lin1(self.single_ln1(si))
        n = self.fourier_embedding(0.25 * torch.log(t_hat / sigma_data))  # (B, c)
        si += self.single_lin2(self.single_ln2(n[:, None]))
        si += self.single_trans1(si)
        si += self.single_trans2(si)
        return si, zij


class Transition(nn.Module):
    """
    Transition
    """

    def __init__(self, in_channel, n=4):
        super(Transition, self).__init__()
        self.ln = nn.LayerNorm(in_channel)
        self.lin1 = nn.Linear(in_channel, in_channel * n, bias=False)
        self.lin2 = nn.Linear(in_channel, in_channel * n, bias=False)
        self.lin3 = nn.Linear(in_channel * n, in_channel, bias=False)

    def forward(self, x):
        """forward"""
        x = self.ln(x)
        a = self.lin1(x)
        b = self.lin2(x)
        x = self.lin3(nn.functional.silu(a) * b)
        return x


class FourierEmbedding(nn.Module):
    """
    FourierEmbedding
    """

    def __init__(self, c):
        super(FourierEmbedding, self).__init__()
        self.w = nn.Parameter(torch.empty(c, dtype=torch.float32))
        nn.init.normal_(self.w)
        self.w.stop_gradient = True
        self.b = nn.Parameter(torch.empty(c, dtype=torch.float32))
        nn.init.normal_(self.b)
        self.b.stop_gradient = True

    def forward(self, t_hat):
        """
        t_hat: (B,)
        return:
            (B, c)
        """
        y = torch.cos(2 * np.pi * (t_hat[:, None] * self.w[None] + self.b[None]))
        return y


class DiffusionTransformer(nn.Module):
    """
    DiffusionTransformer
    """

    def __init__(self,
                 a_channel: int,
                 s_channel: int,
                 z_channel: int,
                 n_block: int,
                 n_head: int):
        super(DiffusionTransformer, self).__init__()
        self.n_block = n_block
        self.n_head = n_head

        self.attention_list = nn.ModuleList()
        self.transition_list = nn.ModuleList()
        for n in range(self.n_block):
            self.attention_list.append(AttentionPairBias(
                a_channel, s_channel, z_channel,
                self.n_head, has_si=True))
            self.transition_list.append(ConditionedTransitionBlock(
                a_channel, s_channel))

    def forward(self, ai, si, zij, beta):
        """forward"""
        for attention, transition in zip(self.attention_list, self.transition_list):
            ai += recompute_wrapper(attention,
                                    ai, si, zij, beta, is_recompute=self.training)
            ai += recompute_wrapper(transition,
                                    ai, si, is_recompute=self.training)
        return ai


class AttentionPairBias(nn.Module):
    """AttentionPairBias"""

    def __init__(self, a_channel, s_channel, z_channel,
                 n_head, has_si, dropout_rate=0.1):
        super(AttentionPairBias, self).__init__()

        self.has_si = has_si
        self.n_head = n_head
        self.head_dim = a_channel // n_head

        if has_si:
            self.ln = AdaLN(a_channel, s_channel)
        else:
            self.ln = nn.LayerNorm(a_channel)
        self.q_lin = nn.Linear(a_channel, a_channel)
        self.k_lin = nn.Linear(a_channel, a_channel, bias=False)
        self.v_lin = nn.Linear(a_channel, a_channel, bias=False)
        self.b_ln = nn.LayerNorm(z_channel)
        self.b_lin = nn.Linear(z_channel, n_head, bias=False)
        self.g_lin = nn.Linear(a_channel, a_channel, bias=False)
        self.alpha_dropout = nn.Dropout(dropout_rate)
        self.out_lin1 = nn.Linear(a_channel, a_channel, bias=False)
        self.out_dropout = nn.Dropout(dropout_rate)
        if has_si:
            self.out_lin2 = nn.Linear(s_channel, a_channel)
            nn.init.constant_(self.out_lin2.bias, -2.0)

        default_M = 10000
        self._AttenIndex = AttentionIndex(max_atom_num=default_M)

    def forward(self, ai, si, zij, beta):
        """
        ai: (B, N, d1)
        si: (B, N, d1)
        zij: (B, N, N, d2)
        beta: (B, N, N) or (1, N, N)
        attention_idx
        """
        assert self.has_si == (not si is None)

        B, N, D = ai.shape
        H, d = self.n_head, self.head_dim

        if self.has_si:
            ai = self.ln(ai, si)
        else:
            ai = self.ln(ai)
        q = self.q_lin(ai).reshape([B, N, H, d]).permute(0, 2, 1, 3)  # (B, H, N, d)
        k = self.k_lin(ai).reshape([B, N, H, d]).permute(0, 2, 1, 3)  # (B, H, N, d)
        v = self.v_lin(ai).reshape([B, N, H, d]).permute(0, 2, 1, 3)  # (B, H, N, d)
        # zij is not tiled by diff_batch_size so far
        b = self.b_lin(self.b_ln(zij))  # (B, N, N, H) or (B, C, nq, nk, H)
        g = nn.functional.sigmoid(self.g_lin(ai)) \
            .reshape([B, N, H, d]).permute(0, 2, 1, 3)  # (B, H, N, d)
        diff_batch_size = ai.shape[0] // b.shape[0]

        if len(zij.shape) == 5:
            # local attention
            M = ai.shape[1]
            atten_idx = self._AttenIndex.get_atten_idx(M)
            atten_idx = {k: v.to(ai.device) for k, v in atten_idx.items()}

            query_idx = atten_idx['query_idx'].flatten()  # [C,32]
            query_mask = atten_idx['query_mask'][..., None, None, None]  # [C,32,1,1,1]
            key_idx = atten_idx['key_idx'].flatten()  # [C,128,1,1,1]
            key_mask = atten_idx['key_mask'][..., None, None, None]  # [C,128,1,1,1]
            alpha_mask = atten_idx['alpha_mask']  # [C,32,128]
            C, n_query, n_key = alpha_mask.shape

            q = q.permute(2, 0, 1, 3)  # (B, H, N, d) -> (N, B, H, d)
            k = k.permute(2, 0, 1, 3)  # (B, H, N, d) -> (N, B, H, d)
            v = v.permute(2, 0, 1, 3)  # (B, H, N, d) -> (N, B, H, d)
            g = g.permute(2, 0, 1, 3)  # (B, H, N, d) -> (N, B, H, d)

            query_like_shape = [C, n_query, B, H, d]
            key_like_shape = [C, n_key, B, H, d]

            query_mask = query_mask.type(q.dtype)
            key_mask = key_mask.type(k.dtype)
            q = q[query_idx].reshape(query_like_shape) * query_mask  # (C, 32, B, H, d)
            k = k[key_idx].reshape(key_like_shape) * key_mask  # (C, 128, B, H, d)
            v = v[key_idx].reshape(key_like_shape) * key_mask  # (C, 128, B, H, d)
            g = g[query_idx].reshape(query_like_shape) * query_mask  # (C, 32, B, H, d)

            q = q.permute(2, 3, 0, 1, 4)  # (C, 32, B, H, d) -> (B, H, C, 32, d)
            k = k.permute(2, 3, 0, 1, 4)  # (C, 128, B, H, d) -> (B, H, C, 128, d)
            v = v.permute(2, 3, 0, 1, 4)  # (C, 128, B, H, d) -> (B, H, C, 128, d)
            g = g.permute(2, 3, 0, 1, 4)  # (C, 32, B, H, d) -> (B, H, C, 32, d)
            b = b.permute(0, 4, 1, 2, 3)  # (b, C, 32, 128, H) -> (b, H, C, 32, 128)

            if diff_batch_size > 1:
                b = tile_batch_dim(b, diff_batch_size)
            beta = tile_batch_dim(beta, ai.shape[0])
            b = (b + beta.unsqueeze(1))  # (B, H, C, 32, 128)
            alpha = torch.matmul(q / np.sqrt(d), k.transpose(-2, -1)) + b  # (B, H, C, 32, 128)
            alpha = torch.nn.functional.softmax(alpha)
            alpha = self.alpha_dropout(alpha)

            ai = torch.matmul(alpha, v) * g  # (B, H, C, 32, d)
            ai = ai.reshape([B, H, C * n_query, d])
            ai = ai[:, :, :si.shape[1], :]

        else:
            # global attention
            if diff_batch_size > 1:
                b = tile_batch_dim(b, diff_batch_size)
            if beta.shape[0] == 1:
                beta = beta.tile([ai.shape[0], 1, 1])
            b = (b + beta[..., None]).permute(0, 3, 1, 2)  # (B, N, N, H) -> (B, H, N, N)

            alpha = torch.matmul(q / np.sqrt(d), k.transpose(-2, -1)) + b  # (B, H, N, N)
            alpha = F.softmax(alpha)  # (B, H, N, N)
            alpha = self.alpha_dropout(alpha)

            ai = torch.matmul(alpha, v) * g  # (B, H, N, d)

        ai = ai.permute(0, 2, 1, 3).reshape([B, N, D])
        ai = self.out_lin1(ai)
        ai = self.out_dropout(ai)
        if self.has_si:
            ai = nn.functional.sigmoid(self.out_lin2(si)) * ai
        return ai


class ConditionedTransitionBlock(nn.Module):
    """
    ConditionedTransitionBlock
    """

    def __init__(self, a_channel, s_channel, n=2):
        super(ConditionedTransitionBlock, self).__init__()
        self.ln = AdaLN(a_channel, s_channel)
        self.lin1 = nn.Linear(a_channel, a_channel * n, bias=False)
        self.lin2 = nn.Linear(a_channel, a_channel * n, bias=False)
        self.lin3 = nn.Linear(s_channel, a_channel)
        nn.init.constant_(self.lin3.bias, -2.0)
        self.lin4 = nn.Linear(a_channel * n, a_channel, bias=False)

    def forward(self, a, s):
        """forward"""
        a = self.ln(a, s)
        b = nn.functional.silu(self.lin1(a)) * self.lin2(a)
        a = nn.functional.sigmoid(self.lin3(s)) * self.lin4(b)
        return a


class AdaLN(nn.Module):
    """
    AdaLN
    """

    def __init__(self, a_channel, s_channel):
        super(AdaLN, self).__init__()
        self.a_ln = nn.LayerNorm(a_channel, elementwise_affine=False)
        self.s_ln = nn.LayerNorm(s_channel)
        self.s_ln.bias = None
        self.lin1 = nn.Linear(s_channel, a_channel)
        self.lin2 = nn.Linear(s_channel, a_channel, bias=False)

    def forward(self, a, s):
        """forward"""
        a = self.a_ln(a)
        s = self.s_ln(s)
        a = nn.functional.sigmoid(self.lin1(s)) * a + self.lin2(s)
        return a


""" Atom Attention """


class AtomAttentionEncoder(nn.Module):
    """
    AtomAttentionEncoder: only support multimer-monomer
    """

    def __init__(self,
                 in_token_channel: int,
                 out_token_channel: int,
                 token_pair_channel: int,
                 atom_channel: int,
                 atom_pair_channel: int,
                 use_dense_mode: bool,
                 atom_transformer: nn.Module):
        super(AtomAttentionEncoder, self).__init__()
        self.config = config
        in_token_channel = in_token_channel
        out_token_channel = out_token_channel
        token_pair_channel = token_pair_channel
        atom_channel = atom_channel
        atom_pair_channel = atom_pair_channel

        self.ap_util = AtomPairUtil()
        self.dense = use_dense_mode

        f_dim = 3 + 1 + 1 + 128 + 4 * 64
        self.lin_atom_meta_to_cond_feat = \
            nn.Linear(f_dim, atom_channel, bias=False)
        self.lin_pos_offset_to_apair = \
            nn.Linear(3, atom_pair_channel, bias=False)
        self.lin_inv_sq_dist_to_apair = \
            nn.Linear(3, atom_pair_channel, bias=False)
        self.lin_valid_mask_to_apair = \
            nn.Linear(1, atom_pair_channel, bias=False)

        # embed trunk single embedding to cond atom feat
        self.ln_trunk_single_to_cond_atom_feat = nn.LayerNorm(in_token_channel)
        self.lin_trunk_single_to_cond_atom_feat = \
            nn.Linear(in_token_channel, atom_channel, bias=False)

        # embed cond pair embedding to pair representation
        self.ln_cond_pair_feat_to_pair_repr = nn.LayerNorm(token_pair_channel)
        self.lin_cond_pair_feat_to_pair_repr = \
            nn.Linear(token_pair_channel, atom_pair_channel, bias=False)

        # embed noise position
        self.lin_noise_pos_to_single_repr = \
            nn.Linear(3, atom_channel, bias=False)

        # embed single cond to pair representation
        self.act_single_cond_to_pair_repr = nn.ReLU(atom_channel)
        self.lin_single_cond_to_pair_repr = nn.Linear(
            atom_channel, atom_pair_channel, bias=False)

        # pair activation MLP
        self.mlp_pair_active = torch.nn.Sequential(
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias=False),
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias=False),
            nn.ReLU(atom_pair_channel),
            nn.Linear(atom_pair_channel, atom_pair_channel, bias=False),
        )

        self.atom_transformer = atom_transformer

        # aggregate atom representation to token representation
        self.act_atom_to_token = nn.ReLU()
        self.lin_atom_to_token = \
            nn.Linear(atom_channel, out_token_channel, bias=False)

    def forward(self, feature, rl, s_trunk, zij):
        """
        Args:
        - rl:         [B, M, 3]
        - s_trunk:    [B, N, c_s=384]
        - zij:        [b, N, N, c_z=128]

        Use features:
        - f_ref_pos:              [B, M, 3]
        - f_ref_charge:           [B, M]
        - f_ref_mask:             [B, M]
        - f_ref_element:          [B, M, 128]
        - f_ref_atom_name_chars:  [B, M, 4, 64]
        - f_ref_space_uid:        [B, M]

        Returns:
        - ai:     [B, N, C_t=768]
        - ql:     [B, M, C_a=128]
        - cl:     [B, M, C_a=128]
        - plm:    [B, M, M, C_ap=16]
        """
        DIFFUSION = rl is not None
        if DIFFUSION:  # late tile zij for avoiding OOM
            diff_batch_size = rl.shape[0] / zij.shape[0]  # B/b

        atom_token_uid = feature['ref_token2atom_idx']  # [B,M]
        atom_mask = torch.ones_like(atom_token_uid, dtype=torch.long)  # [B,M]

        # create teh atom single conditioning: embed per-atom meta data
        f_ref_element = F.one_hot(feature['ref_element'], num_classes=128)  # (B, M, 128)
        f_ref_pos = feature['ref_pos']  # (B, M, 3)
        f_ref_space_uid = feature['ref_space_uid']  # (B, M)
        f_ref_charge = feature['ref_charge'].unsqueeze(-1).type(f_ref_pos.dtype)  # (B, M, 1)
        f_ref_mask = feature['ref_mask'].unsqueeze(-1).type(f_ref_pos.dtype)  # (B, M, 1)
        f_atom_name_chars = F.one_hot(
            feature['ref_atom_name_chars'], num_classes=64)  # (B, M, 4, 64)
        f_atom_name_chars = f_atom_name_chars.reshape(
            [f_atom_name_chars.shape[0], f_atom_name_chars.shape[1], 4 * 64])  # (B, M, 4*64)
        atom_feat_concat = torch.concat(
            [f_ref_pos, f_ref_charge, f_ref_mask, f_ref_element, f_atom_name_chars],
            dim=-1) * f_ref_mask

        cl = self.lin_atom_meta_to_cond_feat(atom_feat_concat)  # (B, M, c_a)

        # embed offsets between atom reference positions
        if DIFFUSION and diff_batch_size > 1:
            f_ref_pos = f_ref_pos[:zij.shape[0]]  # (b, M, 3)
            f_ref_space_uid = f_ref_space_uid[:zij.shape[0]]  # (b, M)

        dlm = self.ap_util.add_2_seqs(f_ref_pos, - f_ref_pos, dense=self.dense)  # [b,C,nq,nk,3]
        vlm = self.ap_util.cmp_2_seqs(f_ref_space_uid, f_ref_space_uid, dense=self.dense)  # [b,C,nq,nk,1]
        vlm = vlm.type(torch.float32)
        plm = self.lin_pos_offset_to_apair(dlm) * vlm  # (b, M, M, c_atompair)

        # embed pairwise inverse squared distancs, and the valid mask
        plm += self.lin_inv_sq_dist_to_apair(1 / (1 + dlm ** 2)) * vlm
        plm += self.lin_valid_mask_to_apair(vlm) * vlm

        # initialize the atom single representation as the single conditioning.
        ql = cl  # (B, M, c_a)

        # if provided, add trunk embedding and noisy positions
        atom_mask = atom_mask.type(ql.dtype)
        if rl is not None:
            # convert s_trunk_tok_i to s_trunk_atom_l
            s_trunk_atom = seq_to_atom_feat(s_trunk, atom_token_uid, atom_mask)  # (B, M, c_s)

            # broadcast the single and pair embedding from the trunk
            cl += self.lin_trunk_single_to_cond_atom_feat(
                self.ln_trunk_single_to_cond_atom_feat(s_trunk_atom)) \
                  * atom_mask.unsqueeze(-1)  # (B, M, c_a)

            zij = self.lin_cond_pair_feat_to_pair_repr(
                self.ln_cond_pair_feat_to_pair_repr(zij))

            plm += self.ap_util.to_atompair(zij=zij, atom_token_uid=atom_token_uid,
                                            atom_mask=atom_mask, dense=self.dense)
            # assert plm.shape[1] == cl.shape[1]

            # Add the noisy positions
            ql += self.lin_noise_pos_to_single_repr(rl) \
                  * atom_mask.unsqueeze(-1)  # (B, M, c_a)

        # add the combined single conditioning to the pair representation
        if DIFFUSION and diff_batch_size > 1:
            single_cond = cl[:zij.shape[0]]  # (b, M, C_a)
        else:
            single_cond = cl  # (B, M, c_a)
        single_cond = self.lin_single_cond_to_pair_repr(
            self.act_single_cond_to_pair_repr(single_cond))  # (b, M, c_atompair)

        single_cond = self.ap_util.add_2_seqs(single_cond, single_cond, dense=self.dense)  # (b, M, M, c_atompair)
        plm += single_cond

        # run MLP on the pair activation
        plm += self.mlp_pair_active(plm)

        # cross attention transformer
        ql = self.atom_transformer(ql, cl, plm)

        # aggregate per-atom representation to per-token representation
        N_token = feature['residue_index'].shape[1]
        al = self.lin_atom_to_token(self.act_atom_to_token(ql))  # (B, M, c_t)
        ai = aggregate_atom_feat_to_token(
            al, atom_token_uid, atom_mask, N_token)  # (B, N_res, c_t)

        return ai, ql, cl, plm


class AtomTransformer(nn.Module):
    " Atom Transformer for HF3. "

    def __init__(self,
                 n_query: int,
                 n_key: int,
                 diffusion_transformer: nn.Module):
        super(AtomTransformer, self).__init__()
        self.n_query = n_query
        self.n_key = n_key
        self.default_size = 10000
        self.diff_transformer = diffusion_transformer
        self._AttenIndex = AttentionIndex(self.default_size, self.n_query, self.n_key)

    def forward(self, ql, cl, plm):
        """
        ql: (B, M, d1)
        cl: (B, M, d1)
        plm: (B, M, M, d2)
        atten_idx: q,k,v,b,g indices for local seq attention
        """

        M = ql.shape[1]
        atten_idx = self._AttenIndex.get_atten_idx(M)
        atten_idx = {k: v.to(ql.device) for k, v in atten_idx.items()}
        if len(plm.shape) == 4:
            # sparse plm
            beta = self._get_beta_mask(M)  # [M,M]
        else:
            # dense plm
            assert len(plm.shape) == 5
            alpha_mask = atten_idx['alpha_mask']
            beta = torch.full_like(alpha_mask, fill_value=-10.0 ** 10, dtype=torch.bfloat16)
            beta[alpha_mask == 1] = 0

        beta = beta[None]  # [1, M, M] or [1, C, 32, 128]
        ql = self.diff_transformer(ql, cl, plm, beta=beta)
        return ql

    def _get_beta_mask(self, M):
        if self.beta is None:
            self.beta = self._gen_beta_mask(self.default_size)

        if M > self.default_size:
            # gen a larger beta mask
            return self._gen_beta_mask(M)

        return self.beta[:M, :M]

    def _gen_beta_mask(self, M):
        subset_centers = self._get_subset_centers(M)
        beta = np.full([M, M], -10.0 ** 10, dtype=np.float32)
        half_width = self.n_query // 2
        half_height = self.n_key // 2

        for c in subset_centers:
            left = np.max([c - half_width, 0])
            right = np.min([c + half_width, M])
            top = np.max([c - half_height, 0])
            bottom = np.min([c + half_height, M])
            beta[left:right, top:bottom] = 0.0

        return torch.tensor(beta)

    def _get_subset_centers(self, M):
        half = (self.n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, self.n_query, dtype=np.float32)
        return np.round(centers).astype(np.int64)


class AtomAttentionDecoder(nn.Module):
    " Atom Attention Decoder for HF3. "

    def __init__(self,
                 in_token_channel: int,
                 atom_channel: int,
                 final_zero_init: bool,
                 atom_transformer: nn.Module):
        super(AtomAttentionDecoder, self).__init__()
        self.config = config
        token_channel = in_token_channel
        out_channel = 3
        atom_channel = atom_channel
        self.lin0 = nn.Linear(token_channel, atom_channel, bias=False)
        self.atom_transformer = atom_transformer
        self.ln1 = nn.LayerNorm(atom_channel)

        if final_zero_init:
            self.lin1 = nn.Linear(atom_channel, out_channel, bias=False)
            nn.init.constant_(self.lin1.weight, 0.)
        else:
            weight_init = None
            self.lin1 = nn.Linear(atom_channel, out_channel, bias=False,
                                  weight=weight_init)

    def forward(self, ai, ql_skip, cl_skip, plm_skip, atom_token_uid, atom_mask):
        """
        Args:
        ai: [B,N_res,C_s]
        ql: [B,M,C_a]
        cl: [B,M,C_a]
        plm: [B,M,M,C_atompair]
        atom_token_uid: [B,M]
        atom_mask: [B,M]

        Returns:
        r_udpate: [B, M, 3]
        """

        # Broadcast per-token activiations to per-atom activations and add the skip connection
        al = seq_to_atom_feat(ai, atom_token_uid, atom_mask)  # (B, M, C_t)
        al = al * atom_mask.unsqueeze(-1)  # (B, M, C_s)
        ql_skip += self.lin0(al) * atom_mask.unsqueeze(-1)  # (B, M, C_a)

        # cross attention transformer
        ql_skip = self.atom_transformer(ql_skip, cl_skip, plm_skip, ) \
                  * atom_mask.unsqueeze(-1)

        # Map to position update
        r_update = self.lin1(self.ln1(ql_skip)) * atom_mask.unsqueeze(-1)  # (B, M, 3)

        return r_update


class RelativePositionEncoding(nn.Module):
    """
    Algorithm 3: RelativePositionEncoding
    """

    def __init__(self, channel_num, config, global_config):
        super(RelativePositionEncoding, self).__init__()
        self.channel_num = channel_num
        self.config = config
        self.global_config = global_config

        self.rel_position_project = nn.Linear(
            # rel_pos & rel_token: R_max * 2 + 2
            # same_entity: 1
            # rel_chain: S_max * 2 + 2
            2 * (self.config.relative_token_max * 2 + 2) + 1 + \
            2 * self.config.relative_chain_max + 2,
            self.channel_num['token_pair_channel'],
            bias=False)

    def forward(self, batch):
        asym_id = batch['asym_id']
        same_chain = asym_id.unsqueeze(dim=-2) == asym_id.unsqueeze(dim=-1)

        pos = batch['residue_index']
        same_residue = pos.unsqueeze(dim=-2) == pos.unsqueeze(dim=-1)

        entity_id = batch['entity_id']
        same_entity = entity_id.unsqueeze(dim=-2) == entity_id.unsqueeze(dim=-1)

        def _calc_clipped_offset(fi, r_max, is_same):
            offset = fi.unsqueeze(dim=-1) - fi.unsqueeze(dim=-2)
            clipped_offset = torch.clip(offset + r_max, min=0, max=2 * r_max)
            final_offset = torch.where(
                is_same,
                clipped_offset,
                (2 * r_max + 1) * torch.ones_like(clipped_offset))
            return nn.functional.one_hot(final_offset, 2 * r_max + 2)

        rel_pos = _calc_clipped_offset(
            pos, self.config.relative_token_max, same_residue)

        token_id = batch['token_index']
        rel_token = _calc_clipped_offset(
            token_id, self.config.relative_token_max,
            torch.logical_and(same_residue, same_chain))

        sym_id = batch['sym_id']
        rel_chain = _calc_clipped_offset(
            sym_id, self.config.relative_chain_max,
            torch.logical_not(same_chain))

        same_entity_ = same_entity.unsqueeze(dimdimdim=-1).type(rel_pos.dtype)
        rel_act = torch.concat(
            [rel_pos, rel_token, same_entity_, rel_chain], dim=-1)
        return self.rel_position_project(rel_act)


def singleton(cls):
    instances = {}

    def get_instance(*args, **kwargs):
        if cls not in instances:
            instances[cls] = cls(*args, **kwargs)
        return instances[cls]

    return get_instance


@singleton
class AttentionIndex:
    " Attention Index for Local Sequence Attention. "

    def __init__(self, max_atom_num=10000, n_query=32, n_key=128):
        self.n_query = n_query
        self.n_key = n_key
        self._key_list = ['query_idx', 'query_mask', 'key_idx', 'key_mask',
                          'alpha_mask', 'pair_idx']
        self._upd_M_N_create_index(max_atom_num)

    def _upd_M_N_create_index(self, M):
        """ Update M and create a larger arange of indices. 

        [IMPORTANT] call order matters
        """
        self.max_atom_num = M  # always update M first
        self._create_subset_centers()
        self._create_local_pair_idx()
        self._create_attention_idx()
        self._prepare_api()

    def _create_subset_centers(self):
        """ Find all subset centers for local sequence attention.

        Create:
        - self._centers: ndarray, shape [C]
        - self.n_centers: int, the number of centers.
        """
        M = self.max_atom_num
        half = (self.n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, self.n_query, dtype=np.float32)
        centers = np.round(centers).astype(np.int32)
        self._centers = centers
        self.n_centers = len(centers)

    def _create_local_pair_idx(self):
        """ Find involved atom-level features' indices for local sequence attention.

        For local sequence attention of a pair of atom-level query ql and key qm, 
        compute the indices of corresponding features for every local attention.

        Create:
        - xid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of query feat in local sequence attention.
        - yid: ndarray, shape: [C*(B-T)*(R-L)]
            The dense indices of key feat in local sequence attention.
        """
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num
        centers, C = self.get_current_centers()
        xid, yid = [], []
        for c in centers:
            T = np.max([c - nq // 2, 0])
            B = np.min([c + nq // 2, M])
            L = np.max([c - nk // 2, 0])
            R = np.min([c + nk // 2, M])
            x, y = np.meshgrid(np.arange(T, B, dtype=np.int64),
                               np.arange(L, R, dtype=np.int64))
            # shape: x:[R-L, B-T], y:[R-L, B-T]
            xid.append(x.T.flatten())  # [(B-T)*(R-L)]
            yid.append(y.T.flatten())  # [(B-T)*(R-L)]
        xid = np.concatenate(xid, axis=0)
        yid = np.concatenate(yid, axis=0)
        self._xid, self._yid = xid, yid

    def _create_attention_idx(self):
        """ Find all relevent indices required by attention.

        Create Indices:
        - query_idx: ndarray [C, n_query]
        - query_mask: ndarray [C, n_query]
        - key_idx: ndarray [C, n_key]
        - key_mask: ndarray [C, n_key]
        - alpha_mask: ndarray [C, n_query, n_key]
        - pair_idx: ndarray [C, n_query, n_key]
        """
        self.attention_idx = None  # clear old index for saving memory
        centers, C = self.get_current_centers()
        nq, nk = self.n_query, self.n_key
        M = self.max_atom_num

        # query indices
        query_idx = np.arange(M, dtype=np.int64)
        query_pad_len = C * nq - M
        query_pad = np.full([query_pad_len], -1, dtype=np.int64)
        query_idx = np.concatenate([query_idx, query_pad]).reshape(C, nq)
        query_mask = (query_idx != -1).astype(np.int64)
        query_idx *= query_mask

        # key indices
        key_blks = []
        for c in centers:
            start = np.max([c - nk // 2, 0])
            end = np.min([c + nk // 2, M])
            window_len = end - start
            key_id_c = np.full(nk, -1, dtype=np.int64)
            key_id_c[:window_len] = np.arange(start, end, dtype=np.int64)
            key_blks.append(key_id_c)
        key_idx = np.concatenate(key_blks).reshape([C, nk])
        key_mask = (key_idx != -1).astype(np.int64)
        key_idx *= key_mask

        # alpha mask
        alpha_mask = query_mask[..., np.newaxis] * key_mask[:, np.newaxis]  # [C,nq,nk]
        valid_alpha_idx = alpha_mask.flatten().nonzero()[0]

        # bias indices and beta indices
        valid_pair_id = self._xid * M + self._yid
        pair_idx = np.zeros_like(alpha_mask, dtype=np.int64).flatten()
        np.put_along_axis(pair_idx, valid_alpha_idx, valid_pair_id, axis=0)
        pair_idx = pair_idx.reshape([C, nq, nk])

        # save
        self._query_idx, self._query_mask = query_idx, query_mask
        self._key_idx, self._key_mask = key_idx, key_mask
        self._alpha_mask = alpha_mask
        self._pair_idx = pair_idx

    def _prepare_api(self):
        """ Convert indices from ndarray to torch tensor. """
        self.xy_idx = (torch.tensor(self._xid), torch.tensor(self._yid))
        self.attention_idx = {'query_idx': torch.tensor(self._query_idx),
                              'query_mask': torch.tensor(self._query_mask),
                              'key_idx': torch.tensor(self._key_idx),
                              'key_mask': torch.tensor(self._key_mask),
                              'alpha_mask': torch.tensor(self._alpha_mask),
                              'pair_idx': torch.tensor(self._pair_idx),
                              'centers': torch.tensor(self._centers)}

    def _slice_subset_index(self, atten_idx, C):
        return copy.deepcopy({k: atten_idx[k][:C] for k in self._key_list})

    def _padding(self, index, M):
        """ Replace out-of-range indices by 0s.
        Args:
        - index: dict[tensor]
        - M: int, new maximum number of atoms.
        """
        for k, m in [('query_idx', 'query_mask'), ('key_idx', 'key_mask')]:
            idx = index[k]
            mask = index[m]
            mask[idx >= M] = 0
            idx[idx >= M] = 0
            index[k], index[m] = idx, mask

        index['alpha_mask'] = \
            index['query_mask'][..., None] * index['key_mask'][:, None]

        C, nq, nk = index['pair_idx'].shape
        xid, yid = self.get_xy_idx(M)
        valid_pair_id = xid * M + yid
        valid_alpha_idx = index['alpha_mask'].flatten().nonzero().squeeze(-1)
        index['pair_idx'].reshape([C * nq * nk])

        # pad valid_pair_id and valid_alpha_idx to length of pair_idx
        valid_pair_id = torch.concat([valid_pair_id,
                                      valid_pair_id[-1].expand(C * nq * nk - valid_pair_id.shape[0])])
        valid_alpha_idx = torch.concat([valid_alpha_idx,
                                        valid_alpha_idx[-1].expand(C * nq * nk - valid_alpha_idx.shape[0])])

        index['pair_idx'].reshape([C * nq * nk]).scatter_(0, valid_alpha_idx, valid_pair_id)
        index['pair_idx'] = index['pair_idx'].reshape([C, nq, nk])

    def get_current_centers(self):
        """ Get currently-using subset center. No update. """
        return self._centers, self.n_centers

    def get_centers(self, M, n_query=32):
        """ Compute the subset centers for any given M. """
        half = (n_query - 1.0) * 0.5
        centers = np.arange(half, M + half, n_query, dtype=np.float32)
        centers = np.round(centers).astype(np.int32)
        return centers, len(centers)

    def get_xy_idx(self, M):
        """ Get query and key indices for any given max number of atoms.

        Returns:
        - x_idx: tensor
        - y_idx: tensor
        """
        assert self.xy_idx is not None
        if M == self.max_atom_num:
            return self.xy_idx
        if M > self.max_atom_num:
            # create_larger index for larger M
            self._upd_M_N_create_index(M)
            return self.xy_idx

        # slice subset indices
        xid, yid = self.xy_idx
        subset_mask = (xid < M) * (yid < M)
        return xid[subset_mask], yid[subset_mask]

    def get_atten_idx(self, M):
        """ Get attention indices for any given max number of atoms.

        [Important] Assume Idx's content won't be changed. Otherwise, return a deepcopy version. 

        Returns:
        attention index: dict[tensor]
        - query_idx: tensor, shape: [C, n_query]
        - query_mask: tensor, shape: [C, n_query]
        - key_idx: tensor, shape: [C, n_key]
        - key_mask: tensor, shape: [C, n_key]
        - alpha_mask: tensor, shape: [C, n_query, n_key]
        - pair_idx: tensor, shape: [C, n_query, n_key]
        - centers: tensor, shape: [C]
        """
        if M == self.max_atom_num:
            return self.attention_idx

        if M > self.max_atom_num:
            # create larger index for larger M
            self._upd_M_N_create_index(M)
            return self.attention_idx

        # slice subset indices
        centers, C = self.get_centers(M)
        sub_atten_idx = self._slice_subset_index(self.attention_idx, C)
        # replace out-of-range indices by zero padding
        self._padding(sub_atten_idx, M)
        for k in self._key_list:
            sub_atten_idx[k] = torch.tensor(sub_atten_idx[k])
        sub_atten_idx['centers'] = torch.tensor(centers)
        return sub_atten_idx


class AtomPairUtil():
    """ Compute atompair features according to atom attention indices. """

    def __init__(self, M=10000, n_query=32, n_key=128):
        self.M = M
        self.nq = n_query
        self.nk = n_key
        self._AttenIdx = AttentionIndex(M, n_query=n_query, n_key=n_key)

    def to_atompair(self, zij, atom_token_uid, atom_mask, dense):
        """ Convert token-level pair feature to atom-pair features. 

        Args:
        - zij: Tensor, [B,N,N,D]
            token-level pair feature
        - atom_token_uid: Tensor, [B,M]
            token to atom unique id.
        - atom_mask: Tensor, [B,M]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if dense:
            return self._pair_to_atompair_dense(zij, atom_token_uid)
        else:
            return self._pair_to_atompair_sparse(zij, atom_token_uid, atom_mask)

    def add_2_seqs(self, ql, qm, dense):
        """ Compute atompair by adding two atom-level sequences feature. 

        Args:
        - ql, qm: Tensor [B,M,D]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if len(ql.shape) == 2:
            ql, qm = ql.unsqueeze(-1), qm.unsqueeze(-1)
        result = self._add_2_seqs_dense(ql, qm) if dense else self._add_2_seqs_sparse(ql, qm)
        return result if len(ql.shape) != 2 else result.squeeze(-1)

    def cmp_2_seqs(self, ql, qm, dense):
        """ Compute atompair by comparing two atom-level sequences feature. 

        Args:
        - ql, qm: Tensor [B,M,D]
        - local: bool, use local attention (return dense feature) 
                        or global attention (return sparse feature)
        """
        if len(ql.shape) == 2:
            ql, qm = ql.unsqueeze(-1), qm.unsqueeze(-1)
        result = self._cmp_2_seqs_dense(ql, qm) if dense else self._cmp_2_seqs_sparse(ql, qm)
        return result

    def _pair_to_atompair_sparse(self, f_pair, atom_token_uid, atom_mask):
        """ Convert per-token pair feature to sparse atom-pair features.

        Args:
        - f_pair:         Tensor: [B,N,N,D]
        - atom_token_uid: Tensor: [B,M]
        - atom_mask:      Tensor: [B,M]

        Returns:
        - f_pair:         Tensor: [B,M,M,D]
        """
        B, N, _, D = f_pair.shape
        M = atom_token_uid.shape[1]
        diff_batch_size = atom_token_uid.shape[0] // B
        if diff_batch_size > 1:
            atom_token_uid = atom_token_uid[:B]
            atom_mask = atom_mask[:B]

        atom_token_uid = atom_token_uid.flatten(0, 1)  # [B*M]
        f_pair = f_pair.flatten(0, 1)  # [B*N,N,D]
        f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1])  # [B*M,N,D]
        f_pair = f_pair.reshape([B, M, N, D]).permute(0, 2, 1, 3).flatten(0, 1)  # [B*N,M,D]
        f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1])  # [B*M,M,D]

        return f_pair.reshape([B, M, M, D]).permute(0, 2, 1, 3)  # [B,M,M,D]

    def _pair_to_atompair_dense(self, f_pair, atom_token_uid):
        """ Convert per-token pair feature to dense atom-pair features.

        Args:
        - f_pair:         Tensor: [B,N,N,D]
        - atom_token_uid: Tensor: [B,M]

        Returns:
        - f_pair:         Tensor: [B,C,nq,nk,D]
        """
        b, _, _, D = f_pair.shape
        M = atom_token_uid.shape[1]
        pair_mask = self._AttenIdx.get_atten_idx(M)['alpha_mask'].to(f_pair.device)  # [C,nq,nk]
        C, nq, nk = pair_mask.shape
        # local indices (x,y) in atompair, dense tensor:[num_valid_ap] ≈ C*nq*nk
        ap_id_x, ap_id_y = self._AttenIdx.get_xy_idx(M)
        valid_pair_idx = pair_mask.flatten().nonzero()  # [num_valid_ap]
        valid_pair_idx = torch.cat([valid_pair_idx,
                                    valid_pair_idx[-1].expand([C * nq * nk - valid_pair_idx.shape[0], -1])],
                                   dim=0)

        ap = []
        for i in range(b):
            uid = atom_token_uid[i]
            # query and key feats
            x_seq_id, y_seq_id = uid[ap_id_x], uid[ap_id_y]
            ap_feat_dense = f_pair[i][x_seq_id, y_seq_id]  # [num_valid_ap, b, D]
            ap_feat_dense = torch.cat([ap_feat_dense,
                                       ap_feat_dense[-1].expand([C * nq * nk - ap_feat_dense.shape[0], -1])],
                                      dim=0)
            atompair = torch.zeros([C * nq * nk, D], dtype=ap_feat_dense.dtype).to(f_pair.device)
            atompair.scatter_(0, valid_pair_idx, ap_feat_dense)
            atompair = atompair.reshape([C, nq, nk, D])  # [B,C,nq,nk,D]
            ap.append(atompair)

        ap = torch.stack(ap)
        return ap

    def _operate_2_seqs(self, ql, qm, func):
        """ Compute atompair from two atom-level sequences feature thru give function.
        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]
        - func: binary function take (ql, qm) to operate

        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        B, M, D = ql.shape
        xid, yid = self._AttenIdx.get_xy_idx(M)  # dense:[num_valid_ap]->C*nq*nk
        pair_mask = self._AttenIdx.get_atten_idx(M)['alpha_mask'].to(ql.device)  # [C,nq,nk]
        C, nq, nk = pair_mask.shape
        valid_pair_idx = pair_mask.flatten().nonzero()  # [num_valid_ap]

        ql = ql.permute(1, 0, 2)  # [M, B, D]
        qm = qm.permute(1, 0, 2)  # [M, B, D]
        ap_dense = func(ql[xid], qm[yid])  # [num_valid_ap, B, D]

        ap_sparse = torch.zeros([C * nq * nk, B, D], dtype=ap_dense.dtype).to(ql.device)

        valid_pair_idx = torch.cat([valid_pair_idx,
                                    valid_pair_idx[-1].expand([C * nq * nk - valid_pair_idx.shape[0], -1])],
                                   dim=0).unsqueeze(-1)
        ap_dense = torch.cat([ap_dense,
                              ap_dense[-1].expand([C * nq * nk - ap_dense.shape[0], -1, -1])],
                             dim=0)

        ap_sparse.scatter_(0, valid_pair_idx, ap_dense)
        return ap_sparse.reshape([C, nq, nk, B, D]).permute(3, 0, 1, 2, 4)  # [B,C,nq,nk,D]

    def _add_2_seqs_sparse(self, ql, qm):
        """ Create sparse feature by adding two atom-level sequences feature.
        Args:
        - ql: Tensor:[B,M,D]
        - qm: Tensor:[B,M,D]

        Returns:
        - ap: Tensor:[B,M,M,D]
        """
        return ql.unsqueeze(2) + qm.unsqueeze(1)

    def _add_2_seqs_dense(self, ql, qm):
        """ Create dense feature by adding two atom-level sequences feature.
        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]

        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        return self._operate_2_seqs(ql, qm, lambda x, y: x + y)

    def _cmp_2_seqs_sparse(self, ql, qm):
        """ Create sparse feature by comparing two atom-level sequences feature.

        Args:
        - ql: Tensor:[B,M,D]
        - qm: Tensor:[B,M,D]

        Returns:
        - ap: Tensor:[B,M,M,D]
        """
        ap = ql.unsqueeze(1) == qm.unsqueeze(2)
        return ap.type("float32").unsqueeze(-1)

    def _cmp_2_seqs_dense(self, ql, qm):
        """ Create dense feature by comparing two atom-level sequences feature.

        Args:
        - ql: Tensor: [B,M,D]
        - qm: Tensor: [B,M,D]

        Returns:
        - ap: Tensor:[B,C,nq,nk,D]
        """
        return self._operate_2_seqs(ql, qm, lambda x, y: (x == y).type(x.dtype))


""" Per-token features to per-atom features """


def seq_to_atom_feat(f_token, atom_token_uid, atom_mask):
    """
    Convert per-token sequence features to per-atom features.
    Args:
        f_token:              [B,N,D]
        atom_token_uid:       [B,M]
        atom_mask:            [B,M]

    Returns:
        f_atom:         [B,M,D]
    """
    B, M = atom_token_uid.shape[:2]
    f_atom = []
    atom_mask = atom_mask.type(f_token.dtype)
    for i in range(B):
        idx = atom_token_uid[i]
        mask = atom_mask[i]
        atom = f_token[i][idx] * mask[..., None]
        f_atom.append(atom.unsqueeze(0))
    f_atom = torch.concat(f_atom, dim=0)
    return f_atom


def pair_to_atompair(f_pair, atom_token_uid, atom_mask):
    """
    Convert per-token pair feature to atom-pair features.

    Args:
    f_pair:         [B,N,N,D]
    atom_token_uid: [B,M]
    atom_mask:      [B,M]

    Returns:
    f_pair:         [B,M,M,D]
    """
    B, N, _, D = f_pair.shape
    M = atom_token_uid.shape[1]
    diff_batch_size = atom_token_uid.shape[0] // B
    if diff_batch_size > 1:
        atom_token_uid = atom_token_uid[:B]
        atom_mask = atom_mask[:B]

    atom_token_uid = atom_token_uid.flatten(0, 1)  # [B*M]
    f_pair = f_pair.flatten(0, 1)  # [B*N,N,D]
    f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1])  # [B*M,N,D]
    f_pair = f_pair.reshape([B, M, N, D]).permute(0, 2, 1, 3).flatten(0, 1)  # [B*N,M,D]
    f_pair = f_pair[atom_token_uid] * atom_mask.reshape([-1, 1, 1])  # [B*M,M,D]

    return f_pair.reshape([B, M, M, D]).permute(0, 2, 1, 3)


""" Per-atom features to per-token features """


def aggregate_atom_feat_to_token(f_atom, atom_token_uid, atom_mask, n_token):
    """ Aggregate per-atom features to per-token features.

    Args:
    f_atom: [B,M,D]
    atom_token_uid: [B,M]
    atom_mask: [B,M]
    n_token: the number of token

    Returns:
    f_token: [B,N,D]
    """
    B, _, D = f_atom.shape[:3]

    f_atom_mean = []
    for i in range(B):
        idx = atom_token_uid[i].type(torch.long)
        mask = atom_mask[i]

        data = f_atom[i][mask == 1]
        segment_ids = idx[mask == 1]
        count_data = mask[mask == 1]

        atom_sum = torch.zeros(segment_ids.max().item() + 1, data.size(1), device=data.device)
        atom_sum.scatter_add_(0, segment_ids.unsqueeze(1).expand(-1, data.size(1)), data)

        atom_count = torch.zeros(segment_ids.max().item() + 1, device=data.device)
        atom_count.scatter_add_(0, segment_ids, count_data)

        atom_mean = atom_sum / (atom_count[:, None] + 1e-8)
        f_atom_mean.append(atom_mean.unsqueeze(0))
    f_atom_mean = torch.concat(f_atom_mean, dim=0)
    pad_len = n_token - f_atom_mean.shape[1]
    padding = torch.zeros([B, pad_len, D], dtype=f_atom.dtype).to(f_atom.device)
    f_token = torch.concat([f_atom_mean, padding], 1)
    return f_token


if __name__ == "__main__":
    import config_tmp as config

    all_config = config.model_config('allatom_demo')
    channel_num = all_config.model.channel_num
    model_config = all_config.model
    global_config = model_config.global_config

    diffusion_module = DiffusionModule(channel_num, model_config.heads.diffusion_module, global_config)
    # print(diffusion_module)

    input_representation = {
        'single_inputs': torch.ones(25, 64, 384 + 32 + 32 + 1),  # the most initial representation
        'single': torch.ones(25, 64, 384),
        'pair': torch.ones(5, 64, 64, 128),
        'rel_pos_encoding': torch.ones(5, 64, 64, 128),
    }

    input_batch = {
        'all_atom_pos_mask': torch.ones(25, 64 * 5),
        'seq_mask': torch.ones(25, 64),
        'ref_token2atom_idx': torch.ones(25, 64 * 5, dtype=torch.int64),
        'ref_element': torch.ones(25, 64 * 5, dtype=torch.int64),
        'ref_pos': torch.ones(25, 64 * 5, 3),
        'ref_space_uid': torch.ones(25, 64 * 5),
        'ref_charge': torch.ones(25, 64 * 5),
        'ref_mask': torch.ones(25, 64 * 5),
        'ref_atom_name_chars': torch.ones(25, 64 * 5, 4, dtype=torch.int64),
        'residue_index': torch.ones(25, 64),
    }

    diffusion_module.cuda()
    input_batch = {k: v.cuda() for k, v in input_batch.items()}
    input_representation = {k: v.cuda() for k, v in input_representation.items()}

    output = diffusion_module.sample_diffusion(input_representation, input_batch, step_num=3)
    print(output)
