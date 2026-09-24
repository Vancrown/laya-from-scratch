import torch
import torch.nn.functional as F


def log_score(proba: torch.Tensor, outcomes: torch.Tensor):
    # -NLL
    outcome_proba = proba.gather(dim=-1, index=outcomes.unsqueeze(-1)).squeeze(-1)
    return torch.log(outcome_proba.clamp_min(1e-12))


def brier_score_reward(proba: torch.Tensor, outcomes: torch.Tensor):
    # -brier score
    targets = F.one_hot(outcomes, num_classes=proba.shape[-1]).to(proba.dtype)

    error2 = (proba - targets).pow(2).sum(dim=-1)
    return -error2


def spherical_score(proba: torch.Tensor, outcomes: torch.Tensor):
    outcome_proba = proba.gather(dim=-1, index=outcomes.unsqueeze(-1)).squeeze(-1)

    norm = torch.linalg.vector_norm(proba, dim=-1).clamp_min(1e-12)
    return outcome_proba / norm


def ranked_proba_score(proba: torch.Tensor, outcomes: torch.Tensor):
    targets = F.one_hot(outcomes, num_classes=proba.shape[-1]).to(proba.dtype)

    pred_cdf = proba.cumsum(dim=-1)
    target_cdf = targets.cumsum(dim=-1)

    err2 = (pred_cdf[..., :-1] - target_cdf[..., :-1]).pow(2).sum(dim=-1)
    return -err2
