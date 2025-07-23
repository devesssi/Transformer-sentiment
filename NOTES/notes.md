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



