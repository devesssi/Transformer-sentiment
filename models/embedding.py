import torch
import torch.nn as nn
import math


# class positionalembed(nn.Module):
#     def __init__(self,d_model):
#         super().__init()__
#         self    
#     def forward(self, x):
#         self

class Embeddings(nn.Module):
    
    def __init__(self, vocabsize, d_model, max_length =5000):
        super().__init__()
        self.embedding = nn.embedding(vocabsize, d_model)
        self.positionalencod = positionalencoding(d_model , max_length)
        
    def forward(self, x):
        x = x.embedding(x)
        x = x.positionalencod(x)
        return x
