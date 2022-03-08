import torch
from torch import nn
from sentence_transformers import SentenceTransformer, util

class NeuralNetwork(nn.Module):
    def __init__(self):
        super(NeuralNetwork, self).__init__()
        self.sim_ = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

    def forward(self, x):
        return x