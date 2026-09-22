import torch
import torch.nn as nn

from laya.transformer import TinyTransformerEncoder


class DecisionHead(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(d_model, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor):
        logits = self.net(x)
        return logits.squeeze(-1)


class TinyDecisionModel(nn.Module):
    def __init__(
        self,
        encoder: TinyTransformerEncoder,
        d_model: int,
        decision_hidden_dim: int,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.encoder = encoder
        self.decision_head = DecisionHead(d_model, decision_hidden_dim)

    def forward(self, token_ids: torch.Tensor, option_mask: torch.Tensor):
        hidden_states = self.encoder(token_ids)

        option_states = hidden_states[option_mask]

        batch_size = token_ids.shape[0]
        num_options = option_mask.sum(dim=1)

        if not torch.all(num_options == num_options[0]):
            raise ValueError(
                "All examples in a batch must have the same number of options."
            )

        option_states = option_states.view(batch_size, num_options[0].item(), -1)
        logits = self.decision_head(option_states)
        proba = torch.softmax(logits, dim=-1)
        return logits, proba
