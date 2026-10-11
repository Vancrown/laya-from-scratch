import torch
import torch.nn as nn

from laya.scoring import log_score


def multi_obj_reward(
    proba: torch.Tensor,
    outcomes: torch.Tensor,
    cost_matrix: torch.Tensor,
    confidence_threshold: float = 0.75,
    scoring_weight: float = 0.5,
    utility_weight: float = 1,
    risk_weight: float = 1,
    review_weight: float = 1,
    review_cost: float = 0.3,
):
    confidences, preds = proba.max(dim=-1)

    auto = confidences >= confidence_threshold
    correct = preds == outcomes

    # score
    scoring_reward = log_score(proba, outcomes)

    # correct
    correctness_reward = (auto & correct).to(proba.device)

    # misclassification risk
    risk_cost = cost_matrix[outcomes, preds]
    risk_cost = risk_cost * auto.to(proba.device)

    # human review
    review = ~auto
    operational_cost = review.to(proba.device) * review_cost

    # combine all
    total_reward = (
        scoring_reward * scoring_weight
        + correctness_reward * utility_weight
        - risk_cost * risk_weight
        - operational_cost * review_weight
    )

    return {
        "total": total_reward,
        "scoring": scoring_reward,
        "correctness": correctness_reward,
        "risk": risk_cost,
        "operation": operational_cost,
    }


def expected_cost_decision(
    proba: torch.Tensor, cost_matrix: torch.Tensor, review_cost: float = 0.3
):
    expected_cost = proba @ cost_matrix
    best_cost, best_action = expected_cost.min(dim=-1)
    should_review = best_cost > review_cost
    return {
        "best_action": best_action,
        "expected_cost": expected_cost,
        "should_review": should_review,
    }
