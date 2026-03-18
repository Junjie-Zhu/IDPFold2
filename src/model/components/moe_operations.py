import torch

def binned_gather(x, indices, tokens_per_expert, expert_capacity):
    num_experts = tokens_per_expert.shape[0]
    num_columns = x.shape[1]

    out = torch.zeros((num_experts, expert_capacity, num_columns),
                      dtype=x.dtype, device=x.device)

    x_gathered = torch.index_select(x, dim=0, index=indices)

    expert_ids = torch.arange(num_experts, device=tokens_per_expert.device).repeat_interleave(tokens_per_expert)
    capacity_indices = torch.cat([torch.arange(c, device=tokens_per_expert.device) for c in tokens_per_expert])
    dest_indices = expert_ids.long() * expert_capacity + capacity_indices.long()
    dest_indices = torch.stack([
        dest_indices,
        torch.arange(dest_indices.shape[0], device=dest_indices.device)
    ], dim=1).squeeze(-1)  # (num_assigned_tokens, 2)
    valid_mask = capacity_indices < expert_capacity

    x_to_scatter = x_gathered[valid_mask]
    dest_indices = dest_indices[valid_mask]

    out_flat = out.view(-1, num_columns)
    out_flat[dest_indices[:, 0], :] = x_to_scatter
    return out, dest_indices


def binned_scatter(
        x: torch.Tensor,
        indices: torch.Tensor,  # original x --> sorted x
        dest_indices: torch.Tensor,  # sorted x --> binned x
        tokens_per_expert: torch.Tensor,
        top_k: int,
        expert_weights: torch.Tensor = None,
) -> torch.Tensor:
    _, _, hidden_size = x.shape
    tokens = tokens_per_expert.sum().item() // top_k  # total tokens

    # use the indices to reversely gather from binned x
    x_out = torch.zeros((tokens, hidden_size),
                      dtype=x.dtype,
                      device=x.device)
    x_sorted = torch.zeros((tokens * top_k, hidden_size),
                           dtype=x.dtype,
                           device=x.device)
    x_flat = x.view(-1, hidden_size)
    x_sorted[dest_indices[:, 1], :] = x_flat[dest_indices[:, 0], :]

    # apply expert weights if provided
    if expert_weights is not None:
        expert_weights = torch.index_select(expert_weights, dim=0, index=indices)  # (tokens,)
        x_sorted = x_sorted * expert_weights[..., None]  # (tokens, hidden_size)

    x_out.index_add_(0, indices, x_sorted)
    return x_out


