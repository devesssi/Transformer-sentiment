1.Why only encoder?
For sentiment analysis, you’re doing sequence-to-label, not sequence-to-sequence.

Input: text sequence

Output: single label (positive/negative)

You don’t need to generate sequences (like in translation, where decoder is required).
Encoder alone can learn a representation of the input sequence and you attach a classifier head on top of it.
That’s why in models like BERT, only the encoder is used.

If you wanted to build:

machine translation → you’d need both encoder + decoder.

text summarization → encoder + decoder (if generative).

question answering → encoder is often enough (depends).


🔷 Why only encoder?
A Transformer has two main parts in the original paper:

Encoder stack

Decoder stack

Encoder:
Takes an input sequence and computes a contextualized representation of the whole sequence.
For each token, it learns a vector that contains information about that token and its relationship to all others.

Decoder:
Takes the encoder’s output and generates an output sequence, one token at a time.
Used in tasks where the output is a sequence, and it needs to depend on what has already been generated (e.g., translation, summarization).

which part belongs to what ?
If something has trainable weights and runs during model(input), it belongs in models/.
If something happens before batching (like tokenization or padding), it belongs in data/ or utils/.

| Phase           | What it does                                                       | Where it belongs    | Why                                            |
| --------------- | ------------------------------------------------------------------ | ------------------- | ---------------------------------------------- |
| **Pre-model**   | Load text, tokenize, pad, convert to token IDs                     | `data/` or `utils/` | It's static logic, not trained                 |
| **Model input** | Token IDs → Embeddings → Positional Encodings → Attention → Output | `models/`           | It runs inside `model(x)` and contains weights |


nn.embeddings-->

🎯 What is nn.Embedding?
It’s a lookup table that maps discrete token IDs (integers) to dense vectors (embeddings).

Think of it as a trainable dictionary:
token_id → vector representation

✅ Signature
python
Copy
Edit
nn.Embedding(
    num_embeddings,  # total number of tokens in your vocab
    embedding_dim    # dimensionality of the vector for each token
)
🧠 What do the inputs mean?
1. num_embeddings → your vocab size
This is the total number of unique tokens you allow.

For example:

Vocab: ["<pad>", "<unk>", "I", "love", "NLP"]

num_embeddings = 5

Each token gets an index from 0 to 4, and nn.Embedding(5, embedding_dim) stores 5 learnable vectors.

2. embedding_dim → vector size for each token
This is how “deep” the representation for each word will be. Typical values:

128 (lightweight)

256 / 512 (standard)

768 (BERT)

1024 (big)

Each token ID will be mapped to a vector of shape [embedding_dim].

