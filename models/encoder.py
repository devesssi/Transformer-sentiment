"""
Encoder Module
==============
This module implements the Transformer encoder as described in "Attention Is All You Need".

Process Flow:
-------------
1. Embeddings:
   - Input tokens are integer IDs from the vocabulary.
   - The `Embeddings` class maps each ID to a dense vector of size `d_model`.

2. Positional Encoding:
   - Since self-attention is order-agnostic, we add a deterministic sine/cosine positional encoding.
   - This allows the model to understand token order in the sequence.

3. Encoder Blocks:
   - The encoder contains `N` stacked `EncoderBlock`s.
   - Each block has:
       a) Multi-Head Self-Attention: lets each token gather context from all others.
       b) Feed Forward Network: applies a nonlinear transformation per token.
       c) Residual Connections: allow gradient flow and stabilize learning.
       d) Layer Normalization: keeps activations numerically stable.

4. Final Normalization:
   - After all encoder blocks, a `LayerNorm` is applied to normalize the final output.

Input & Output:
---------------
- Input: `src` (shape [batch_size, seq_len]) — token IDs.
- Output: Tensor of shape [batch_size, seq_len, d_model] with context-enriched token embeddings.

Arguments:
----------
- vocab_size: Size of vocabulary (number of unique tokens).
- d_model: Dimensionality of embeddings & model hidden states.
- N: Number of encoder blocks to stack.
- heads: Number of attention heads in each block.
- d_ff: Hidden layer size in feedforward network inside each block.
- dropout: Dropout probability for regularization.
"""


import torch
import torch.nn as nn

from models.embedding import Embeddings
from models.embedding import positionalEncoding
from models.encoder_block import EncoderBlock



class Encoder(nn.Module):
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout):
        super().__init__()
        self.embed = Embeddings(vocab_size, d_model)
        self.pe = positionalEncoding(d_model)
        self.layers = nn.ModuleList([EncoderBlock(...) for _ in range(N)])
        self.norm = nn.LayerNorm(d_model)

    def forward(self, src, mask=None):
        x = self.embed(src)
        x = self.pe(x)
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)
