# How it works step-by-step
# Input x → shape [batch, seq_len, d_model].

# MultiHeadAttention takes (Q=K=V=x) for self-attention.

# Add residual: x + attn_out so gradients flow and original info is preserved.

# LayerNorm normalizes per feature so training is stable.

# FeedForward applies two Linear layers with ReLU between → learns non-linear features per token.

# Add another residual: x + ffn_out.

# Final LayerNorm.,FFN (feedforward network) is applied to the output of the attention layer.
#shapes matter a lot in transformers, so be careful with them. dff is the dimension of the feedforward network, which is usually larger than d_model(4 times).
# The output shape is [batch, seq_len, d_model] after the final linear layer.
# self.linear1 = nn.Linear(d_model, d_ff)

# Weight shape: (d_ff, d_model).

# Input x shape: [B, L, d_model]. PyTorch applies linear on the last dim → result [B, L, d_ff].

# self.linear2 = nn.Linear(d_ff, d_model)

# Projects back: [B, L, d_ff] -> [B, L, d_model].
# Why expand to d_ff? Gives the network capacity to learn complex, higher-dimensional interactions (like a bottleneck architecture reversed).

# Important: Because the same nn.Linear modules are applied to the last dimension, the operation is position-wise — tokens don’t exchange info here.


# Dropout is a regularization technique used in neural networks to prevent overfitting by randomly
# setting a fraction of neurons to zero during the training phase, forcing the network to learn more robust and generalized features
# rather than relying on specific neurons or their co-adaptations.
#  This process effectively creates an ensemble of smaller,
# dynamically changing sub-networks during training, which enhances the model's ability to generalize to unseen data.
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.attention import MultiHeadAttention

class PositionalwiseFeedForward(nn.Module):
    def __init__(self,d_model,d_ff,dropout= 0.1):
        super.__init__()
        self.linear1 = nn.linear(d_model, d_ff) #Expand 
        self.linear2 = nn.linear(d_ff, d_model) #Project back
        self.dropout = nn.Dropout(dropout)
    def forward(self, x):
        return self.linear2(self.Droupout(F.relu(self.linear1(x))))
    
    
class EncoderBlock(nn.Module):
    def __init__(self, d_model, heads, d_ff, dropout=0.1):
        super().__init__()
        self.mha = MultiHeadAttention(heads, d_model)  # your existing class
        self.ffn = PositionalwiseFeedForward(d_model, d_ff, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        # Multi-head attention + residual + norm
        attn_out, _ = self.mha(x, x, x, mask)  
        x = self.norm1(x + self.dropout1(attn_out))

        # Feed forward + residual + norm
        ffn_out = self.ffn(x)
        x = self.norm2(x + self.dropout2(ffn_out))

        return x


# ## **1. Residual Connections**

# Also called **skip connections**, they were popularized by ResNet.

# ### **The problem they solve**

# When you stack many layers in a deep network:

# * Gradients get **smaller** and **vanish** while backpropagating.
# * Or they get **too big** and explode.
# * Training becomes **unstable** and slow.

# ### **The idea**

# Instead of passing data only through a complicated transformation $F(x)$,
# you *add* the original input back:

# $$
# y = x + F(x)
# $$

# Here:

# * $x$ = original input to the layer
# * $F(x)$ = transformation (e.g., attention or feed-forward network)
# * $x + F(x)$ → **short path** for gradients to flow directly from output back to input during training.

# Think of it like a **highway** for gradients:
# Even if $F(x)$ is bad early in training, $x$ can still pass through, preventing information loss.

# ---

# ✅ **Benefits:**

# 1. Solves vanishing gradient problem.
# 2. Makes deep models trainable.
# 3. Allows layers to focus on *refining* features, not *relearning* them from scratch.

# ---

# ## **2. Layer Normalization**

# Also called **LayerNorm**.

# ### **The problem it solves**

# During training:

# * Different batches or time steps may have activations of very different scales.
# * This **internal covariate shift** slows learning and destabilizes the model.

# ### **The idea**

# For **each token vector** in the sequence:

# * Compute its **mean** and **variance** across all features.
# * Normalize so mean = 0, variance = 1.
# * Then scale & shift with learnable parameters ($\gamma$, $\beta$) so the model can still represent needed distributions.

# Mathematically:

# $$
# \text{LayerNorm}(x) = \gamma \cdot \frac{x - \mu}{\sigma} + \beta
# $$

# Where:

# * $\mu$ = mean of features in that token vector
# * $\sigma$ = std deviation of features
# * $\gamma$, $\beta$ = learnable scaling and bias.

# ---

# **Key difference from BatchNorm:**

# * **BatchNorm** normalizes **across batch** (depends on other samples in the batch).
# * **LayerNorm** normalizes **across features** for **each token individually** (independent of batch size).
# * LayerNorm works better for NLP because sequence length and batch sizes vary.

# ---

# 💡 **Together:**

# * **Residuals** → let info/gradients skip past complex layers.
# * **LayerNorm** → keeps activations numerically stable and consistent, so optimization works better.

