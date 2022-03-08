from torch.utils.data import DataLoader
import torch
from torch import nn
import torch.nn.functional as ff
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime
import itertools
from textutil import split_into_sentences, K_sim_scores_single
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')

from torch_model.dataset import FakeNewsDataset
from util import *
from baseline import NeuralNetwork
from sentence_transformers import CrossEncoder, SentenceTransformer

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

softmax = lambda x: np.exp(x) / sum(np.exp(x))

# Pre-processing
stances = pd.read_csv('combined_stances_train.csv')
bodies = pd.read_csv('combined_bodies_train.csv')

# Remove unrelated rows
stances = stances.drop(stances[stances.Stance == 'unrelated'].index)

train_set = FakeNewsDataset()
train_set.set_df(stances, bodies)

nli_model = CrossEncoder('cross-encoder/nli-distilroberta-base')
sim_model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')
label_mapping = ['contradiction', 'entailment', 'neutral']

N = 100 # len(train_set)
S = np.zeros((N,3))
y = np.zeros((N,))

for i in range(N):
    print(f'{i}/{N}')
    (head, body), label = train_set.__getitem__(i)
    body_sentences = split_into_sentences(body)
    # sim_idx should give the indices of most similary body_sentences
    sim_idx, sim_scores = K_sim_scores_single(sim_model, head, body_sentences)
    # only check nli scores for similar sentences
    nli_scores = nli_model.predict(list(itertools.product([head], body_sentences)))
    # weighting nli_scores by sim_scores to aggregate
    nli_weighted_avg = np.dot(softmax(sim_scores), nli_scores)
    print(nli_weighted_avg)

    S[i] = nli_weighted_avg
    y[i] = label

print('sshape:', S.shape)
print('yshape:', y.shape)

with open('model/basic_nli/S.npy', 'wb') as f:
    np.save(f, S)

with open('model/basic_nli/y.npy', 'wb') as f:
    np.save(f, y)



# X_train, X_dev, y_train, y_dev = train_test_split(S, y, test_size=0.33, random_state=42)
# clf = LogisticRegression(random_state=0).fit(X_train, y_train)
# acc = clf.score(X_dev, y_dev)
# print(f'acc: {acc}')