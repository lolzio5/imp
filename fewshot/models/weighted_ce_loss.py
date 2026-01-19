import torch


def log_sum_exp(value, weights, dim=None, keepdim=False):
    """
    Numerically stable implementation of:
        log(sum(weights * exp(value), dim))
    """
    if dim is None:
        raise ValueError("dim must be specified")

    m, _ = torch.max(value, dim=dim, keepdim=True)
    value0 = value - m
    out = m + torch.log(torch.sum(weights * torch.exp(value0),
                                  dim=dim, keepdim=True))
    if not keepdim:
        out = out.squeeze(dim)
    return out


def class_select(logits, target):
    """
    Select logits corresponding to target classes.
    Equivalent to: logits[torch.arange(B), target]
    """
    batch_size = logits.size(0)
    device = logits.device
    return logits[torch.arange(batch_size, device=device), target]


def weighted_loss(logits, targets, weights):
    """
    Args:
        logits: [B, C]
        targets: [B]
        weights: [B, C]
    Returns:
        loss per example: [B]
    """
    logsumexp = log_sum_exp(logits, weights, dim=1, keepdim=False)
    selected = class_select(logits, targets)
    return -selected + logsumexp
