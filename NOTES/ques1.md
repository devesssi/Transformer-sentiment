Questions you should ask yourself to grow:

What happens if I stack more encoder layers? What tradeoffs arise?

What is the impact of the number of heads in multi-head attention?

How does sequence length affect memory and computation?

Why is attention better than RNNs? In which situations does it struggle?

What does masking do in attention? Why/when is it required?

What are the limitations of positional encoding?

How does the Transformer handle variable sequence lengths?

1️⃣ Why does the attention mechanism scale better than RNNs?
2️⃣ What would happen if you removed the scaling by sqrt(d k)
3️⃣ Why do we use softmax instead of something else?
4️⃣ What does the mask actually do during training and inference?
5️⃣ Why do we return the attn weights?
6️⃣ Why do Q, K, V come from the same input sequence in self-attention but different sequences in encoder-decoder attention?