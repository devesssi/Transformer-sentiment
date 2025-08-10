import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k):
        super().__init__()
        self.d_k = d_k

    def forward(self, q, k, v, mask=None):
        # 1. Compute raw attention scores
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)

        # 2. Apply mask (if any)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        # 3. Softmax to get attention weights
        attention_weights = F.softmax(scores, dim=-1)

        # 4. Multiply weights by values
        output = torch.matmul(attention_weights, v)

        return output, attention_weights


class MultiHeadAttention(nn.Module):
    def __init__(self, heads, d_model):
        super().__init__()
        assert d_model % heads == 0, "d_model must be divisible by heads"

        self.d_k = d_model // heads
        self.heads = heads

        # Linear layers for Q, K, V
        self.linear_q = nn.Linear(d_model, d_model)
        self.linear_k = nn.Linear(d_model, d_model)
        self.linear_v = nn.Linear(d_model, d_model)

        # Scaled Dot-Product Attention
        self.scaled_dot_product_attention = ScaledDotProductAttention(self.d_k)

        # Final output linear layer
        self.linear_out = nn.Linear(d_model, d_model)

    def forward(self, q, k, v, mask=None):
        batch_size = q.size(0)

        # 1. Linear projections + reshape for multi-heads
        q = self.linear_q(q).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        k = self.linear_k(k).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)
        v = self.linear_v(v).view(batch_size, -1, self.heads, self.d_k).transpose(1, 2)

        # 2. Apply scaled dot-product attention
        output, attention_weights = self.scaled_dot_product_attention(q, k, v, mask)

        # 3. Concatenate heads
        output = output.transpose(1, 2).contiguous().view(batch_size, -1, self.heads * self.d_k)

        # 4. Final linear layer
        output = self.linear_out(output)

        return output, attention_weights


# Note : The .view and .transpose operations are used to rearrange the tensor dimensions for multi-head attention.
#.view is used to reshape the tensor.