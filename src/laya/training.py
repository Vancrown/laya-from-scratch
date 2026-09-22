import torch
import torch.nn as nn


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    token_ids: torch.Tensor,
    option_mask: torch.Tensor,
    targets: torch.Tensor,
):
    model.train()

    optimizer.zero_grad()

    logits, _ = model(token_ids, option_mask)

    # min entropy
    loss = nn.functional.cross_entropy(logits, targets)

    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def predict(model: nn.Module, token_ids: torch.Tensor, option_mask: torch.Tensor):
    model.eval()

    _, proba = model(token_ids, option_mask)
    return proba
