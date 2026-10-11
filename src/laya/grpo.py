import torch
import torch.nn as nn

from laya.pretrained_rlcd import (
    generate_candidate,
    candidate_probabilities,
    decision_reward,
)


def gaussian_log_prob(base_logits, candidates, valid_option_mask, exploration_std):
    finite_logits = base_logits.masked_fill(~valid_option_mask, 0)
    safe_candidates = candidates.masked_fill(~valid_option_mask.unsqueeze(0), 0)
    distribution = torch.distributions.Normal(loc=finite_logits, scale=exploration_std)
    element_log_prob = distribution.log_prob(safe_candidates)
    element_log_prob = element_log_prob.masked_fill(~valid_option_mask.unsqueeze(0), 0)

    return element_log_prob.sum(dim=-1)


@torch.no_grad()
def generate_rollout(
    model,
    input_ids,
    attention_mask,
    option_marker_mask,
    outcomes,
    n_candidates=32,
    exploration_std=0.2,
    scoring_weight=0.5,
):
    model.eval()

    old_logits, _, valid_option_mask = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_marker_mask=option_marker_mask,
    )

    candidates = generate_candidate(
        old_logits, valid_option_mask, n_candidates, exploration_std
    )

    proba = candidate_probabilities(candidates)

    candidate_outcomes = outcomes.unsqueeze(0).expand(n_candidates, -1)

    rewards = decision_reward(proba, candidate_outcomes, scoring_weight)
    advantages = rewards - rewards.mean(dim=0, keepdim=True)

    old_log_proba = gaussian_log_prob(
        old_logits, candidates, valid_option_mask, exploration_std
    )
    return {
        "candidates": candidates,
        "advantages": advantages,
        "rewards": rewards,
        "old_log_proba": old_log_proba,
        "valid_option_mask": valid_option_mask,
    }


def clipped_policy_loss(new_log_proba, old_log_proba, advantages, clip_epsilon=0.2):
    log_ratio = new_log_proba - old_log_proba
    ratio = torch.exp(log_ratio)

    unclipped = ratio * advantages
    clipped = torch.clamp(ratio, 1 - clip_epsilon, 1 + clip_epsilon) * advantages

    loss = -torch.minimum(unclipped, clipped).mean()
    return loss


def gaussian_kl(current_logits, reference_logits, valid_option_mask, exploration_std):
    current = current_logits.masked_fill(~valid_option_mask, 0)
    reference = reference_logits.masked_fill(~valid_option_mask, 0)

    squared_diff = (current - reference).pow(2)

    kl_per_example = squared_diff.sum(dim=-1) / (2 * exploration_std**2)
    return kl_per_example.mean()


def grpo_step(
    model,
    reference_model,
    optimizer,
    input_ids,
    attention_mask,
    option_marker_mask,
    rollout,
    exploration_std=0.2,
    clip_epsilon=0.2,
    kl_beta=0.01,
):
    model.eval()

    current_logits, _, valid_option_mask = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_marker_mask=option_marker_mask,
    )
    new_log_proba = gaussian_log_prob(
        current_logits, rollout["candidates"], valid_option_mask, exploration_std
    )
    policy_loss = clipped_policy_loss(
        new_log_proba, rollout["old_log_proba"], rollout["advantages"], clip_epsilon
    )

    with torch.no_grad():
        reference_logits, _, _ = reference_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            option_marker_mask=option_marker_mask,
        )

    kl = gaussian_kl(
        current_logits, reference_logits, valid_option_mask, exploration_std
    )
    loss = policy_loss + kl_beta * kl

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item(), policy_loss.item(), kl.item()
