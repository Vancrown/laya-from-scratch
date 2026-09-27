import torch
import torch.nn as nn

from .scoring import log_score


def generate_candidates(
    base_logits: torch.Tensor, n_candidates: int, exploration_std: float
):
    noise = (
        torch.randn(n_candidates, *base_logits.shape, device=base_logits.device)
        * exploration_std
    )

    candidate_logits = base_logits.detach().unsqueeze(0) + noise
    return candidate_logits


def candidate_probabilities(candidate_logits: torch.Tensor):
    return torch.softmax(candidate_logits, dim=-1)


def score_candidates(
    candidate_probabilities: torch.Tensor, outcome: torch.Tensor
) -> torch.Tensor:
    n_candidates = candidate_probabilities.shape[0]

    candidate_outcomes = outcome.expand(n_candidates, -1)

    rewards = log_score(candidate_probabilities, candidate_outcomes)

    return rewards


def contextual_rlcd_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    option_mask: torch.Tensor,
    outcome: torch.Tensor,
    n_candidates: int = 16,
    exploration_std: float = 0.5,
):
    base_logits, _ = model(token_ids=token_ids, option_mask=option_mask)

    candidates = generate_candidates(base_logits, n_candidates, exploration_std)
    probabilities = candidate_probabilities(candidates)
    rewards = score_candidates(probabilities, outcome)

    advantages = rewards - rewards.mean(dim=0, keepdim=True)

    distribution = torch.distributions.Normal(loc=base_logits, scale=exploration_std)
    log_probabilities = distribution.log_prob(candidates).sum(dim=-1)

    loss = -(advantages.detach() * log_probabilities).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()


def contextual_reward_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    option_masks: torch.Tensor,
    outcome: torch.Tensor,
    reward_fn,
    n_candidates: int = 16,
    exploration_std: float = 0.5,
):
    base_logits, _ = model(token_ids=token_ids, option_mask=option_masks)

    candidates = generate_candidates(base_logits, n_candidates, exploration_std)

    proba = candidate_probabilities(candidates)

    candidate_outcomes = outcome.expand(n_candidates, -1)

    rewards = reward_fn(proba, candidate_outcomes)

    advantages = rewards - rewards.mean(dim=0, keepdim=True)

    distribution = torch.distributions.Normal(loc=base_logits, scale=exploration_std)
    log_proba = distribution.log_prob(candidates).sum(dim=-1)

    loss = -(advantages.detach() * log_proba).mean()

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
