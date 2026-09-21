import math
import torch
import torch.nn as nn


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, causal: bool = False):
        super().__init__()

        if d_model % n_heads != 0:
            raise ValueError("d_model must be divisible by n_heads")

        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.causal = causal

        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.out_proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        batch_size, seq_len, _ = x.shape

        q = self.q_proj[x]
        k = self.k_proj[x]
        v = self.v_proj[x]

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2.0 - 1)
        scores = scores / math.sqrt(self.head_dim)

        if self.causal:
            mask = torch.triu(
                torch.ones(seq_len, seq_len, device=x.device, dtype=torch.bool),
                diagonal=1,
            )
            scores = scores.masked_fill(mask, float("-inf"))

        attention = torch.softmax(scores, dim=-1)

        out = attention @ v
        out = out.transpose(1, 2).contiguous()

        out = out.view(batch_size, seq_len, self.d_model)
        out = self.out_proj(out)
        if return_attention:
            return out, attention
        return out
