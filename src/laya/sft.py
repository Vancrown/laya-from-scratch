import torch
import torch.nn as nn

import torch.nn.functional as F


def sft_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    option_marker_mask: torch.Tensor,
    targets: torch.Tensor,
):
    model.train()

    logits, _, _ = model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        option_marker_mask=option_marker_mask,
    )

    loss = F.cross_entropy(logits, targets)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


@torch.no_grad()
def evaluate(model, input_ids, attention_mask, option_marker_mask, targets):
    model.eval()

    logits, proba, _ = model(input_ids, attention_mask, option_marker_mask)
    loss = F.cross_entropy(logits, targets)

    predictions = logits.argmax(dim=-1)

    accuracy = (predictions == targets).float().mean()

    return loss.item(), accuracy.item(), proba
