import torch
import torch.nn

import torch.nn.functional as F

from laya.multi_objective_reward import multi_obj_reward


def accuracy(proba: torch.Tensor, targets: torch.Tensor):
    preds = proba.argmax(dim=-1)
    return (preds == targets).float().mean()


def nll(proba: torch.Tensor, targets: torch.Tensor):
    target_proba = proba.gather(dim=-1, index=targets.unsqueeze(-1)).squeeze(-1)

    return -torch.log(target_proba.clamp_min(1e-12)).mean()


def brier_score(proba: torch.Tensor, targets: torch.Tensor):
    target_one_hot = F.one_hot(targets, num_classes=proba.shape[-1]).to(proba.device)

    return (proba - target_one_hot).pow(2).sum(dim=-1).mean()


def expected_calibration_error(
    proba: torch.Tensor, targets: torch.Tensor, n_bins: int = 10
):
    confidences, preds = proba.max(dim=-1)

    correctness = (preds == targets).float()

    boundaries = torch.linspace(0, 1, n_bins + 1, device=proba.device)

    ece = torch.zeros((), device=proba.device)

    for i in range(n_bins):
        lower = boundaries[i]
        upper = boundaries[i + 1]
        if i == 0:
            in_bin = (confidences >= lower) & (confidences <= upper)
        else:
            in_bin = (confidences > lower) & (confidences <= upper)

        if not in_bin.any():
            continue

        bin_accuracy = correctness[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()

        bin_weight = in_bin.float().mean()

        ece += bin_weight * (bin_accuracy - bin_confidence).abs()

    return ece


def decision_metrics(
    proba: torch.Tensor,
    targets: torch.Tensor,
    cost_matrix: torch.Tensor,
    confidence_threshold: float = 0.75,
    review_cost: float = 0.3,
    correct_reward: float = 1,
):
    confidences, preds = proba.max(dim=-1)

    auto = confidences >= confidence_threshold
    review = ~auto

    correct = preds == targets
    wrong_auto = auto & ~correct

    risk_cost = cost_matrix[targets, preds] * auto.to(proba.dtype)

    realized_utility = (
        correct_reward * (auto & correct_reward).to(proba.dtype)
        - risk_cost
        - review_cost * review.to(proba.dtype)
    )

    return {
        "utility": realized_utility.mean(),
        "review_rate": review.float().mean(),
        "wrong_auto_rate": wrong_auto.float().mean(),
        "risk_cost": risk_cost.mean(),
    }


def eval_proba(
    proba: torch.Tensor,
    targets: torch.Tensor,
    cost_matrix: torch.Tensor,
    confidence_threshold: float = 0.75,
    review_cost: float = 0.3,
):
    pred_res = {
        "accuracy": accuracy(proba, targets),
        "nll": nll(proba, targets),
        "brier": brier_score(proba, targets),
        "ece": expected_calibration_error(proba, targets),
    }
    decision_res = decision_metrics(
        proba, targets, cost_matrix, confidence_threshold, review_cost
    )

    res = {**pred_res, **decision_res}
    return {k: v.item() for k, v in res.items()}
