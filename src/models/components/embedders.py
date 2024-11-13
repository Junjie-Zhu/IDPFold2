from functools import partial
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.components.primitives import LinearNoBias


class RelativePositionEncoding(nn.Module):
    """
    Implements Algorithm 3 in AF3
    """

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
            in_features=(4 * self.r_max + 2 * self.s_max + 7), out_features=self.c_z
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
            input_feature_dict["asym_id"][..., :, None]
            == input_feature_dict["asym_id"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_residue = (
            input_feature_dict["residue_index"][..., :, None]
            == input_feature_dict["residue_index"][..., None, :]
        ).long()  # [..., N_token, N_token]
        b_same_entity = (
            input_feature_dict["entity_id"][..., :, None]
            == input_feature_dict["entity_id"][..., None, :]
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
        d_chain = torch.clip(
            input=input_feature_dict["sym_id"][..., :, None]
            - input_feature_dict["sym_id"][..., None, :]
            + self.s_max,
            min=0,
            max=2 * self.s_max,
        ) * b_same_entity + (1 - b_same_entity) * (
            2 * self.s_max + 1
        )  # [..., N_token, N_token]
        a_rel_chain = F.one_hot(d_chain, 2 * (self.s_max + 1))

        if self.training:
            p = self.linear_no_bias(
                torch.cat(
                    [a_rel_pos, a_rel_token, b_same_entity[..., None], a_rel_chain],
                    dim=-1,
                ).float()
            )  # [..., N_token, N_token, 2 * (self.r_max + 1)+ 2 * (self.r_max + 1)+ 1 + 2 * (self.s_max + 1)] -> [..., N_token, N_token, c_z]
            return p
        else:
            del d_chain, d_token, d_residue, b_same_chain, b_same_residue
            origin_shape = a_rel_pos.shape[:-1]
            Ntoken = a_rel_pos.shape[-2]
            a_rel_pos = a_rel_pos.reshape(-1, a_rel_pos.shape[-1])
            chunk_num = 1 if Ntoken < 3200 else 8
            a_rel_pos_chunks = torch.chunk(
                a_rel_pos.reshape(-1, a_rel_pos.shape[-1]), chunk_num, dim=-2
            )
            a_rel_token_chunks = torch.chunk(
                a_rel_token.reshape(-1, a_rel_token.shape[-1]), chunk_num, dim=-2
            )
            b_same_entity_chunks = torch.chunk(
                b_same_entity.reshape(-1, 1), chunk_num, dim=-2
            )
            a_rel_chain_chunks = torch.chunk(
                a_rel_chain.reshape(-1, a_rel_chain.shape[-1]), chunk_num, dim=-2
            )
            start = 0
            p = None
            for i in range(len(a_rel_pos_chunks)):
                data = torch.cat(
                    [
                        a_rel_pos_chunks[i],
                        a_rel_token_chunks[i],
                        b_same_entity_chunks[i],
                        a_rel_chain_chunks[i],
                    ],
                    dim=-1,
                ).float()
                result = self.linear_no_bias(data)
                del data
                if p is None:
                    p = torch.empty(
                        (a_rel_pos.shape[-2], self.c_z),
                        device=a_rel_pos.device,
                        dtype=result.dtype,
                    )
                p[start : start + result.shape[0]] = result
                start += result.shape[0]
                del result
            del a_rel_pos, a_rel_token, b_same_entity, a_rel_chain
            p = p.reshape(*origin_shape, -1)
            if p.shape[-2] > 2000:
                torch.cuda.empty_cache()
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
