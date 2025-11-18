import torch

def binned_gather(x, indices, bins, expert_capacity, top_k):
    num_experts = bins.shape[0]
    out = torch.zeros((num_experts, expert_capacity, x.shape[1]), dtype=x.dtype, device=x.device)

    num_columns = x.shape[1]
    source_indices_flat = indices.long() // top_k
    source_indices_flat = torch.clamp(source_indices_flat, min=0, max=x.shape[0] - 1)

    x_gathered = torch.index_select(x, dim=0, index=source_indices_flat)
    num_assigned_tokens = x_gathered.shape[0]
    dest_indices = torch.arange(num_assigned_tokens, device=out.device, dtype=torch.long)

    out_flat = out.view(-1, num_columns)
    out_flat[dest_indices] = x_gathered
    return out


def binned_scatter(
        x_expert_output: torch.Tensor,
        indices: torch.Tensor,
        top_k: int,
        original_shape: torch.Size = None,  # (Total_Tokens, NUM_COLUMNS)
) -> torch.Tensor:
    if original_shape is None:
        raise ValueError("Must provide original_shape (Total_Tokens, NUM_COLUMNS) for output initialization.")

    NUM_COLUMNS = original_shape[1]
    TOTAL_TOKENS = original_shape[0]
    out = torch.zeros(original_shape, dtype=x_expert_output.dtype, device=x_expert_output.device)

    dest_indices_flat = indices.long() // top_k
    dest_indices_flat = torch.clamp(dest_indices_flat, min=0, max=TOTAL_TOKENS - 1)

    x_to_scatter = x_expert_output.view(-1, NUM_COLUMNS)
    out.index_add_(
        dim=0,
        index=dest_indices_flat,
        source=x_to_scatter
    )
    return out

