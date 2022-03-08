import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# dummy encoder
ENCODE = lambda x: torch.Tensor([[[1, 2, 3]]]) # extra brackets to match (N, S, E) dimensions, see attention docs

class AttnNet(nn.Module):
    '''
    Toy model for attention mechanism
    '''
    def __init__(self, num_heads):
        super(AttnNet, self).__init__()
        self.attention = torch.nn.MultiheadAttention(3, num_heads)
        self.encode = ENCODE

    def forward(self, head: str, body: str):
        head_emb = self.encode(head)
        body_embs = self.encode(body)
        attn_output, attn_weights = self.attention(head_emb, body_embs, body_embs)
        print(attn_weights)
        return attn_output

m = AttnNet(num_heads=1)
out = m('this is a head', 'this is a body sentence')
print(out)