import torch
import torch.nn as nn

from .scoring import log_score


def rlcd_step(
    policy: nn.Module,
    optimizer: torch.optim.Optimizer,
    outcome: torch.Tensor,
    n_candidates: int = 16,
    exploration_std: float = 0.5,
):
    base_logits = policy.logits

    noise = (
        torch.randn(n_candidates, base_logits.shape[-1], device=base_logits.device)
        * exploration_std
    )

    candidate_logits = base_logits.detach() + noise

    candidate_proba = torch.softmax(candidate_logits, dim=-1)

    candidate_outcomes = outcome.expand(n_candidates)

    rewards = log_score(candidate_proba, candidate_outcomes)

    advantages = rewards - rewards.mean()

    distribution = torch.distributions.Normal(loc=base_logits, scale=exploration_std)
    log_proba = distribution.log_prob(candidate_logits).sum(dim=-1)

    loss = -(advantages.detach() * log_proba).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
