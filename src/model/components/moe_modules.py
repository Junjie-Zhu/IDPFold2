import numpy as np
import torch
import torch.nn as nn

import megablocks.ops as ops
from src.model.protein_transformer import TransitionADALN

_LOAD_BALANCING_LOSS = []


def save_load_balancing_loss(loss):
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.append(loss)


def get_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    return _LOAD_BALANCING_LOSS


def clear_load_balancing_loss():
    global _LOAD_BALANCING_LOSS
    _LOAD_BALANCING_LOSS.clear()


def batched_load_balancing_loss(moe_loss_weight, num_layers, num_experts, top_k):
    if moe_loss_weight == 0:
        return 0.0

    tokens_per_expert, expert_scores = zip(*get_load_balancing_loss())

    tokens = expert_scores[0].shape[0]
    expert_scores = torch.cat(expert_scores, dim=1).float()

    if tokens != 0:
        expert_scores = expert_scores.mean(dim=0)
    else:
        expert_scores = expert_scores.sum(dim=0)
    tokens_per_expert = torch.cat(tokens_per_expert).to(expert_scores.dtype)

    scale_numerator = num_experts * moe_loss_weight
    scale_denominator = num_layers * tokens * top_k
    scale = scale_numerator / scale_denominator
    return scale * torch.dot(tokens_per_expert, expert_scores)


class MoE(nn.Module):
    def __init__(
        self, 
        n_experts, 
        n_activated_experts, 
        dim, 
        dim_cond, 
        dim_router_cond=0,
        expansion_factor=4,
        capacity_factor=1.25,
        normalize_expert_weights=True,
        uniform_expert_assignment=False,
        training=True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        self.dim_router_cond = dim_router_cond

        self.shared_expert = TransitionADALN(dim=dim, dim_cond=dim_cond, expansion_factor=expansion_factor)
        self.experts = Experts(n_experts, n_activated_experts, capacity_factor, dim, dim_cond, expansion_factor, training=training)

        self.gate = nn.Sequential(
            nn.Linear(dim + dim_router_cond, n_experts - 1, bias=False),
            nn.Softmax(dim=-1),
        )
        self.training = training

    def forward(self, x, cond, mask, router_condition=None):
        scores, expert_weights, expert_indices = self.router(x, router_condition)

        x_shared = self.shared_expert(x, cond, mask)

        if self.normalize_expert_weights:
            x = (x_shared + self.experts(x, cond, mask, scores, expert_weights, expert_indices) * self.n_activated_experts) / (self.n_activated_experts + 1)
        else:
            x = x_shared + self.experts(x, cond, mask, scores, expert_weights, expert_indices)
        return x
    
    def router(self, x, router_condition=None):
        if router_condition is not None and self.dim_router_cond > 0:
            x = torch.cat([x, router_condition], dim=-1)

        scores = self.gate(x.view(-1, x.shape[-1]))
        expert_weights, expert_indices = self._top_k(scores)
        if self.normalize_expert_weights:
            expert_weights = expert_weights / expert_weights.sum(dim=-1, keepdim=True)

        expert_indices = self._uniform_expert_assignment(expert_indices) if self.uniform_expert_assignment else expert_indices
        return scores, expert_weights, expert_indices

    def _top_k(self, scores):
        if self.n_activated_experts == 1:
            return scores.max(dim=-1, keepdim=True)
        return torch.topk(scores, k=self.n_activated_experts, dim=-1)

    def _uniform_expert_assignment(self, expert_indices):
        out = torch.arange(expert_indices.numel(), dtype=expert_indices.dtype, device=expert_indices.device)
        out = torch.remainder(out, self.n_experts)
        return out.view(expert_indices.shape)


class Experts(nn.Module):
    def __init__(self, n_experts, n_activated_experts, capacity_factor, dim, dim_cond, expansion_factor=4, training=True):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = n_activated_experts
        self.expert_capacity = capacity_factor

        self.expert = nn.ModuleList(
            TransitionADALN(dim=dim, dim_cond=dim_cond, expansion_factor=expansion_factor)
            for _ in range(n_experts)
        )
        
        self.sort_end_bit = max(1, int(np.ceil(np.log2(self.n_experts))))
        self.training = training

    def forward(self, x, cond, mask, scores, expert_weights, top_experts):
        b, n, d = x.shape

        x, tokens_per_expert = self._single_forward(x, cond, mask, expert_weights, top_experts)

        # load balancing loss
        if self.training:
            save_load_balancing_loss((tokens_per_expert, scores))

        x = x.view(b, n, d)
        return x

    def _single_forward(self, x, cond, mask, expert_weights, top_experts):
        expert_weights = expert_weights.flatten()
        top_experts = top_experts.flatten()
        with torch.no_grad():
            indices, bin_ids, bins, tokens_per_expert = (
                self.indices_and_bins(top_experts))

            # If expert_capacity is set to zero, set the number of tokens
            # per expert to the maximum we need to avoid dropping tokens.
            sl, bs, hs = x.size()
            expert_capacity = self.expert_capacity(sl * bs)
            expert_capacity = min(torch.max(tokens_per_expert).item(), expert_capacity)

        x = self.permute_and_compute(
            x,
            cond, 
            mask,
            indices,
            expert_weights,
            bins,
            expert_capacity,
            self.top_k)
        return x, tokens_per_expert

    def expert_capacity(self, tokens):
        return int(self.capacity_factor * self.top_k * tokens / self.n_experts)

    def indices_and_bins(self, top_expert):
        top_expert = top_expert.int()
        bin_ids, indices = ops.sort(top_expert, self.sort_end_bit)
        tokens_per_expert = ops.histogram(top_expert, self.n_experts)

        # Calculate the bin bounds for the sorted tokens.
        bins = ops.inclusive_cumsum(tokens_per_expert, 0)
        bins = bins.view(1) if not len(bins.size()) else bins
        return indices, bin_ids, bins, tokens_per_expert

    def permute_and_compute(
            self,
            x,
            cond,
            mask,
            indices,
            expert_weights,
            bins,
            expert_capacity,
            top_k):
        # Route the tokens for MoE computation.
        x = x.view(-1, x.shape[-1])
        cond = cond.view(-1, cond.shape[-1])
        mask = mask.view(-1, 1)

        x = ops.binned_gather(
            x, indices, bins, expert_capacity, top_k)
        cond = ops.binned_gather(
            cond, indices, bins, expert_capacity, top_k)
        mask = ops.binned_gather(
            mask, indices, bins, expert_capacity, top_k)

        # Perform the expert computation.
        x_out = []
        for i in range(self.n_experts):
            x_i = x[i, :, :]
            cond_i = cond[i, :, :]
            mask_i = mask[i, :, 0].squeeze(-1)
            x_out.append(self.expert[i](x_i, cond_i, mask_i))
        x_out = torch.cat(x_out, dim=0)

        # Un-route the data for the MoE output.
        return ops.binned_scatter(
            x_out, indices, expert_weights, bins, top_k)






