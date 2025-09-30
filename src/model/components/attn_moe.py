import torch
import torch.nn as nn

from src.model.protein_transformer import TransitionADALN


class MoE(nn.Module):
    def __init__(
        self, 
        n_experts, 
        n_activated_experts, 
        dim, 
        dim_cond, 
        expansion_factor=4,
        normalize_expert_weights=True,
        uniform_expert_assignment=True,
    ):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts
        self.normalize_expert_weights = normalize_expert_weights
        self.uniform_expert_assignment = uniform_expert_assignment

        self.experts = nn.ModuleList([
            TransitionADALN(dim=dim, dim_cond=dim_cond, expansion_factor=expansion_factor)
            for _ in range(n_experts)
        ])  # expert_0 is shared, others are routed

        self.gate = nn.Sequential([
            nn.Linear(dim, n_experts - 1, bias=False),
            nn.Softmax(dim=-1),
        ])

    def forward(self, x, cond, mask):
        scores, expert_weights, expert_indices = self.router(x)

    
    def router(self, x, router_condition=None):
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
    def __init__(self, n_experts, n_activated_experts, dim, dim_cond, expansion_factor=4):
        super().__init__()
        self.n_experts = n_experts
        self.n_activated_experts = n_activated_experts

        self.expert = TransitionADALN(dim=dim, dim_cond=dim_cond, expansion_factor=expansion_factor)

    def forward(self, x, cond, mask):
        b, n, d = x.shape

        return self.expert(x, cond, mask)






