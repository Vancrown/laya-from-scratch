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

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        q = q.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.n_heads, self.head_dim).transpose(1, 2)

        scores = q @ k.transpose(-2, -1)
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


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor):
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, causal: bool, *args, **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.norm1 = nn.LayerNorm(d_model)

        self.attention = MultiHeadSelfAttention(d_model, n_heads, causal)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff = FeedForward(d_model, d_ff)

    def forward(self, x: torch.Tensor, return_attention: bool = False):
        normed = self.norm1(x)

        if return_attention:
            attn_out, attention = self.attention(normed, return_attention=True)
        else:
            attn_out = self.attention(normed)

        x = x + attn_out
        x = x + self.ff(self.norm2(x))

        if return_attention:
            return x, attention

        return x


class TinyTransformerEncoder(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        n_layers: int,
        causal: bool = False,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Embedding(max_seq_len, d_model)

        self.layers = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff, causal) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(d_model)

    def forward(self, token_ids: torch.Tensor, return_attention: bool = False):
        batch_size, seq_len = token_ids.shape

        position = torch.arange(seq_len, device=token_ids.device)

        x = (
            self.token_embedding(token_ids)
            + self.position_embedding(position)[None, :, :]
        )

        attentions = []

        for layer in self.layers:
            if return_attention:
                x, attention = layer(x, return_attention=True)
                attentions.append(attention)
            else:
                x = layer(x)

        x = self.norm(x)

        if return_attention:
            return x, attentions
        return x
