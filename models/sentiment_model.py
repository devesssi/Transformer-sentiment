'''''
    What is Pooling
Pooling is a technique used to reduce the spatial dimensions of data, commonly applied in various fields like deep learning and resource management. In deep learning, particularly within convolutional neural networks (CNNs), pooling is a fundamental operation used to downsample feature maps, reducing their size while retaining essential information.
 This process helps in controlling model complexity, reducing overfitting, and improving computational efficiency by lowering the number of parameters and computations required in subsequent layers.
 Pooling achieves this by aggregating information from nearby pixels or feature values into a single representative value using methods like selecting the maximum value (max pooling), calculating the average (average pooling), or computing the root mean square (L2 pooling) within a defined region, known as the pooling window.
 The stride, which defines how far the window moves at each step, also influences the output size.

---

```python
import torch
import torch.nn as nn
from models.encoder import Encoder
```

* **`import torch`** → Loads PyTorch library (core tensor operations, autograd, GPU support, etc.).
* **`import torch.nn as nn`** → Short alias for PyTorch's neural network modules — gives access to `nn.Module`, `nn.Linear`, `nn.LayerNorm`, etc.
* **`from models.encoder import Encoder`** → Imports your custom Transformer Encoder implementation from `encoder.py`. This will handle embeddings, positional encoding, and stacked encoder blocks.

---

```python
class TransformerSentimentClassifier(nn.Module):
```

* **Class definition** → This is your main model for **sentiment analysis** using a Transformer encoder.
* **Inherits from `nn.Module`** → Makes it compatible with PyTorch's training utilities (`.to(device)`, `.parameters()`, etc.).

---

```python
    def __init__(self, vocab_size, d_model, N, heads, d_ff, dropout, num_classes):
        super().__init__()
```

* **`__init__` method** → Constructor for setting up the layers.
* **Arguments**:

  * `vocab_size` → Number of unique tokens in your dataset (used by embedding layer).
  * `d_model` → Dimensionality of the token embeddings and encoder outputs.
  * `N` → Number of encoder blocks to stack.
  * `heads` → Number of attention heads in multi-head attention.
  * `d_ff` → Hidden layer size of the feed-forward network inside encoder blocks.
  * `dropout` → Dropout probability for regularization.
  * `num_classes` → Number of output classes (for IMDB sentiment: 2 → positive/negative).
* **`super().__init__()`** → Calls parent constructor to properly set up `nn.Module`.

---

```python
        self.encoder = Encoder(vocab_size, d_model, N, heads, d_ff, dropout)
```

* Creates your **Transformer Encoder** using the parameters you passed.
* This encoder:

  * Embeds tokens
  * Adds positional encodings
  * Passes data through N encoder blocks (multi-head attention + feed-forward layers)
* Output shape after encoder: `[batch_size, seq_len, d_model]`

---

```python
        self.fc = nn.Linear(d_model, num_classes)  # classification layer
```

* A **fully connected (dense) layer** that takes the encoder output (vector size `d_model`) and maps it to `num_classes`.
* Example: If `d_model=512` and `num_classes=2`, this layer has shape `[512 → 2]`.

---

```python
    def forward(self, src, mask=None):
```

* **`forward` method** → Defines how the model processes input during training/inference.
* **Arguments**:

  * `src` → Tokenized input batch (`[batch_size, seq_len]`).
  * `mask` → Optional attention mask (to block padding tokens).

---

```python
        x = self.encoder(src, mask)  # [batch, seq_len, d_model]
```

* Passes input tokens and mask into the Transformer Encoder.
* Output: one vector of size `d_model` for each token in the sequence.

---

```python
        x = x.mean(dim=1)            # mean pooling
```

* **Mean pooling across sequence length**: Takes the average of token embeddings for each sample.
* Why? Because sentiment is a sequence-level classification task, not per-token — we condense all token vectors into a **single fixed-size vector** `[batch_size, d_model]`.

---

```python
        return self.fc(x)            # [batch, num_classes]
```

* Passes pooled vector into the final classifier.
* Output: `[batch_size, num_classes]` logits (unnormalized probabilities for each class).
* For IMDB, each row has 2 values → one for "negative", one for "positive".
 '''''
 
import torch
import torch.nn as nn
from models.encoder import Encoder

class TransformerSentimentClassifier(nn.Module):
    def __init__(self,d_model, d_ff, vocab_size,N, heads, dropout,num_classes):
        super().__init__()
        self.encoder = Encoder(vocab_size,d_model, N , heads, d_ff, dropout)
        self.fc = nn.Linear(d_model, num_classes) #classification Layer fc--> fully conneted
    def forward(self,src, mask=None):
        x = self.encoder(src, mask)
        x = x.mean(dim=1) #mean pooling
        return self.fc(x)
    
    

'''''
Naively put — the **`fc`** layer is just the **final decision-maker**.

You’ve already processed the review through embeddings, positional encoding, encoder blocks, and pooling, so now you have **one single summary vector** for the whole sentence.

The `fc` layer:

* Takes that vector.
* Multiplies it by some learned numbers.
* Adds some learned offsets.
* Spits out **two numbers** (for IMDB: one for “negative”, one for “positive”).
* The bigger number decides the sentiment.

It’s like the judge at the end of a talent show — all the earlier layers do the heavy work of understanding the performance, and the judge just gives the final **score**.




### **What is `fc`?**

```python
self.fc = nn.Linear(d_model, num_classes)
```

* `fc` stands for **fully connected** (a.k.a. dense) layer.
* It’s essentially a **matrix multiplication** + **bias addition**.
* Input shape: `[batch_size, d_model]`
* Output shape: `[batch_size, num_classes]`

---

### **Mathematics of `nn.Linear`**

If

* **`W`** is the weight matrix of shape `[num_classes, d_model]`
* **`b`** is the bias vector of shape `[num_classes]`
* **`x`** is the input vector from pooling of shape `[batch_size, d_model]`

Then:

$$
\text{output} = x \cdot W^T + b
$$

* **`x · W^T`** → Projects your d\_model-dimensional vector down to `num_classes` dimensions.
* **`+ b`** → Adds a bias term for each output neuron.

---

### **Why is it needed here?**

* After mean pooling, you have a **single vector** that represents the whole movie review.
* But this vector is still in the `d_model` space (e.g., 512 dimensions).
* You want a **classification decision**:

  * For IMDB, **2 classes** → Positive (1) or Negative (0).
  * So you project `[512] → [2]` using `nn.Linear`.

---

### **How training works for it**

* During backpropagation, the weights `W` and biases `b` are updated so that the output logits match the correct labels.
* Over time, this layer learns **decision boundaries** in the `d_model` space for your classes.

---

### **Example**

Say:

* `d_model = 4`
* `num_classes = 2`
* Input pooled vector: `[0.5, -1.2, 0.3, 0.8]`

If weights & bias are:

$$
W = \begin{bmatrix}
0.2 & -0.5 & 0.1 & 0.4 \\
-0.3 & 0.8 & -0.6 & 0.2
\end{bmatrix}
,\quad b = [0.1, -0.2]
$$

Then:

1. Multiply → `[0.5, -1.2, 0.3, 0.8] × W^T`
2. Add `b`
3. Output → `[logit_for_class0, logit_for_class1]` → softmax → probabilities.


'''''