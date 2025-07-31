import torch
import torch.nn as nn
import math


class positionalembed(nn.Module):
    def __init__(self,d_model, max_length= 5000):
        super().__init__()
        self.d_model = d_model
        self.max_length = max_length
        pe = torch.zeros(max_length, d_model)
        position = torch.arrange(0, max_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.arrange(0, d_model ,2)*(math.log(10000.0)/d_model)
        pe[:,0::2] = torch.sin(position * div_term)
        pe[: ,1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        x = x +self.pe[ : , :x.size(1), :]
        return x

class Embeddings(nn.Module):
    
    def __init__(self, vocabsize, d_model, max_length =5000):
        super().__init__()
        self.embedding = nn.embedding(vocabsize, d_model)
        self.positionalencod = positionalembed(d_model , max_length)
        
    def forward(self, x):
        x = x.embedding(x)
        x = x.positionalencod(x)
        return x
