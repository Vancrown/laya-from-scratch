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
        option_mask: torch.Tensor,
    ):
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)

        hidden_states = outputs.last_hidden_state

        option_states = hidden_states[option_mask]

        batch_size = input_ids.shape[0]

        num_options = option_mask.sum(dim=1)

        if not torch.all(num_options == num_options[0]):
            raise ValueError(
                "All examples must currently " "have the same number of options."
            )

        option_states = option_states.view(batch_size, num_options[0].item(), -1)

        logits = self.decision_head(option_states).squeeze(-1)

        proba = torch.softmax(logits, dim=-1)

        return logits, proba
