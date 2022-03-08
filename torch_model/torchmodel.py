import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sentence_transformers import SentenceTransformer
from nltk.tokenize import sent_tokenize
import math

K_PERC=0.25
THETA=0.4

class RelatedNet(nn.Module):
    '''
    Classifies head+body as related or unrelated
    '''
    def __init__(self, k_perc=K_PERC, theta=THETA):
        super(RelatedNet, self).__init__()
        self.sentence_encoder = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
        self.k_perc = k_perc
        self.theta = theta

    def forward(self, head, body):
        body_sents = sent_tokenize(body)
        head_vec = self.sentence_encoder.encode(head)
        
        scores = np.zeros(len(body_sents))
        # Get similarity between head and each sentence in body
        for i, body_sent in enumerate(body_sents):
            body_sent_vec = self.sentence_encoder.encode(body_sent)
            scores[i] = np.dot(head_vec, body_sent_vec)
        
        # Return the average of the top K similarity scores
        k = max(1, math.floor(len(body)*self.k_perc))
        top_k_scores = np.sort(scores)[-k:]

        res = np.mean(top_k_scores) > self.theta
        return [int(res), int(not res)] # [1, 0] if score > theta, else [0, 1]