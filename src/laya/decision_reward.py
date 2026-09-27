import torch
import torch.nn as nn


def downstream_reward(
    proba: torch.Tensor,
    outcomes: torch.Tensor,
    confidence_threshold: float = 0.75,
    correct_reward: float = 1,
    wrong_reward: float = -4,
    review_reward: float = -0.3,
):
    confidences, predictions = proba.max(dim=-1)
    auto = confidences > confidence_threshold
    correct = predictions == outcomes
    auto_reward = torch.where(correct, correct_reward, wrong_reward)
    rewards = torch.where(auto, auto_reward, review_reward)

    return rewards
