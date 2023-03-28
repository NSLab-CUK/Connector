import torch
import torch.nn as nn
import math


class Attention(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(query, key, value, mask=None, dropout=None):
        score = torch.matmul(query, key.transpose(-2, -1) / math.sqrt(query.size(-1)))

        if mask is not None:
            score = score.masked_fill(mask == 0, -torch.inf)

        attn = torch.softmax(score, dim=-1)

        if dropout is not None:
            attn = dropout(attn)

        return torch.matmul(attn, value), attn


class MultiHeadAttention(nn.Module):
    def __init__(self, n_heads, d_model, dropout=0.15):
        super().__init__()

        self.dropout = nn.Dropout(p=dropout)

        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)

        self.attention = Attention()

    def forward(self, query, key, value, mask=None):
        batch_size = query.shape[0]

        query = self.q(query).view(batch_size, -1, self.n_heads, self.d_k)
        key = self.k(key).view(batch_size, -1, self.n_heads, self.d_k)
        value = self.v(value).view(batch_size, -1, self.n_heads, self.d_k)

        x, attn = self.attention(query, key, value, mask, self.dropout)

        x = x.reshape(batch_size, -1, self.h * self.d_k)

        return x
