import torch
from torch import Tensor


def sum_log_probs(left_logits: Tensor, right_logits: Tensor) -> Tensor:
    """Compute log P(sum=z) for z=0..18 using all valid digit combinations."""
    digits = torch.arange(10, device=left_logits.device)
    right_digits = torch.arange(19, device=left_logits.device)[:, None] - digits
    valid = (right_digits >= 0) & (right_digits < 10)
    # For a fixed sum z, each candidate left digit a determines b = z - a.
    joint = (
        left_logits.log_softmax(dim=-1)[:, None, :]
        + right_logits.log_softmax(dim=-1)[:, right_digits.clamp(0, 9)]
    )
    return torch.logsumexp(joint.masked_fill(~valid, -torch.inf), dim=-1)
