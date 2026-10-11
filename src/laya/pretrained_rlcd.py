import torch
import torch.nn as nn

from .decision_reward import downstream_reward
from .scoring import log_score


def generate_candidate(base_logits, valid_option_mask, n_candidates, exploration_std):
    noise = (
        torch.randn(
            n_candidates,
            *base_logits.shape,
            device=base_logits.device,
            dtype=base_logits.dtype,
        )
        * exploration_std
    )

    finite_logits = base_logits.masked_fill(~valid_option_mask, 0)

    candidates = finite_logits.detach().unsqueeze(0) + noise
    candidates = candidates.masked_fill(~valid_option_mask.unsqueeze(0), float("-inf"))

    return candidates


def candidate_probabilities(candidate_logits):
    return torch.softmax(candidate_logits, dim=-1)


def decision_reward(probabilities, outcomes, scoring_weight=0.5):
    utility = downstream_reward(
        probabilities,
        outcomes,
        confidence_threshold=0.75,
        correct_reward=1,
        wrong_reward=-4,
        review_reward=-0.3,
    )
    proper_score = log_score(probabilities, outcomes)

    return utility + proper_score * scoring_weight


def pretrain_rlcd_step(
    model,
    optimizer,
    input_ids,
    attention_mask,
    option_marker_mask,
    outcomes,
    n_candidates=16,
    exploration_std=0.2,
    scoring_weight=0.5,
):
    model.train()

    # forward
    base_logits, _, valid_option_mask = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_marker_mask=option_marker_mask,
    )

    # candidate logits
    candidates = generate_candidate(
        base_logits, valid_option_mask, n_candidates, exploration_std
    )

    # convert to probabilities
    proba = candidate_probabilities(candidates)

    # reward
    candidate_outcomes = outcomes.unsqueeze(0).expand(n_candidates, -1)

    with torch.no_grad():
        rewards = decision_reward(
            proba, candidate_outcomes, scoring_weight=scoring_weight
        )
        advantages = rewards - rewards.mean(dim=0, keepdim=True)

    # gussian policy
    finite_logits = base_logits.masked_fill(~valid_option_mask, 0)
    distribution = torch.distributions.Normal(loc=finite_logits, scale=exploration_std)

    safe_candidates = candidates.masked_fill(~valid_option_mask.unsqueeze(0), 0)

    element_log_prob = distribution.log_prob(safe_candidates)
    element_log_prob = element_log_prob.masked_fill(~valid_option_mask.unsqueeze(0), 0)

    log_proba = element_log_prob.sum(dim=-1)

    # policy loss
    loss = -(advantages.detach() * log_proba).mean()

    # train
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), rewards.mean().item()
