import torch
import torch.nn as nn
from transformers import AutoModel


class PretrainedDecisionModel(nn.Module):
    def __init__(self, model_name: str, decision_hidden_dim: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.backbone = AutoModel.from_pretrained(model_name)

        hidden_size = self.backbone.config.hidden_size
        self.decision_head = nn.Sequential(
            nn.Linear(hidden_size, decision_hidden_dim),
            nn.GELU(),
            nn.Linear(decision_hidden_dim, 1),
        )

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        option_marker_mask: torch.Tensor,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state

        option_states, valid_option_mask = self._gather_option_states(
            hidden_states, option_marker_mask
        )

        logits = self.decision_head(option_states).squeeze(-1)
        logits = logits.masked_fill(~valid_option_mask, float("-inf"))

        proba = torch.softmax(logits, dim=-1)

        return logits, proba, valid_option_mask

    def _gather_option_states(self, hidden_states, option_marker_mask):

        batch_size = hidden_states.shape[0]
        option_counts = option_marker_mask.sum(dim=1)
        max_options = int(option_counts.max().item())
        hidden_size = hidden_states.shape[-1]
        option_states = hidden_states.new_zeros(batch_size, max_options, hidden_size)

        valid_option_mask = torch.zeros(
            batch_size, max_options, dtype=torch.bool, device=hidden_states.device
        )

        for batch_idx in range(batch_size):
            states = hidden_states[batch_idx, option_marker_mask[batch_idx]]

            n_options = states.shape[0]
            option_states[batch_idx, :n_options] = states
            valid_option_mask[batch_idx, :n_options] = True

        return option_states, valid_option_mask
