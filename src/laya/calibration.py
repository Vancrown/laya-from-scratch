import torch
import torch.nn as nn
import torch.nn.functional as F


def accuracy(proba: torch.Tensor, targets: torch.Tensor):
    pred = proba.argmax(dim=-1)
    return (pred == targets).float().mean()


def neg_log_likelihood(proba: torch.Tensor, targets: torch.Tensor):
    target_proba = proba[torch.arange(targets.shape[0]), targets]
    return -torch.log(target_proba.clamp_min(1e-12)).mean()


def brier_score(proba: torch.Tensor, targets: torch.Tensor):
    target_on_hot = F.one_hot(targets, num_classes=proba.shape[-1]).float()

    error2 = (proba - target_on_hot).pow(2)

    return error2.sum(dim=-1).mean()


def expected_calibration_error(
    proba: torch.Tensor,
    targets: torch.Tensor,
    n_bins: int = 10,
):
    confidences, predicts = proba.max(dim=-1)

    correct = (predicts == targets).float()

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

        bin_accuracy = correct[in_bin].mean()
        bin_confidence = confidences[in_bin].mean()
        bin_weight = in_bin.float().mean()

        ece += bin_weight * torch.abs(bin_accuracy - bin_confidence)
    return ece
