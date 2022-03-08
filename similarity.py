from pygame import K_0
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
from datetime import datetime

import warnings
warnings.filterwarnings('ignore')

from torch_model.dataset import FakeNewsDataset
from baseline import NeuralNetwork
from textutil import split_into_sentences

from sentence_transformers import SentenceTransformer, util

BATCH_SIZE=10

# Pre-processing
stances = pd.read_csv('combined_stances_train.csv')
bodies = pd.read_csv('combined_bodies_train.csv')
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((stances['Headline'].values, bodies['articleBody'].values), axis=None))

# load dataset
training_data = FakeNewsDataset('train_stances.csv', 'train_bodies.csv')
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

# Convert stances to related-unrelated
rel_un = stances['Stance'].copy()
n_train = len(rel_un)
for i in range(n_train):
    if rel_un[i] != 'unrelated':
        rel_un[i] = 'related'

# Tokenize Bodies, 
docs = bodies['articleBody']

# Construct body embeddings (currently creates embeddings of a subset)
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

N_bodies = 1000
print(f'Extracting embeddings for {N_bodies} bodies')
def embed_bodies(bodies_set):
    body_embeds = []
    n_bodies = len(bodies_set['articleBody'])
    for body_ID in np.arange(0,N_bodies):
    #for body_ID in range(n_bodies):
        sen_splits = split_into_sentences(bodies['articleBody'][body_ID])
        if len(sen_splits) == 0:
            sen_splits = bodies['articleBody'][body_ID]
        body_embeds.append(model.encode(sen_splits))
    return body_embeds

t0 = datetime.now()
body_embs = embed_bodies(bodies)
t1 = datetime.now()
print(f'Time required for constructing embeddings of {N_bodies} bodies',t1-t0)

with open(f'body_embs_1-{N_bodies}', 'wb') as f:
    np.save(f, embed_bodies)

# Get scores from HuggingFace Sentence-Bert


def mean_K_max(scores,K):
    scores.sort(reverse=True)
    k_largest = scores[0:K]
    return np.mean(k_largest)

def sorted_K_max(scores, K):
    sorted_idx = np.argsort(scores)
    return sorted_idx[:K], scores[sorted_idx]

def K_sim_scores_single(query, body_sentences, k_perc=0.25):
    '''
    Given a stance and body embeddings, return a list of
    of the k_perc% most semantically similar body sentences
    '''
    query_emb = np.array([1, 2, 3]) # model.encode(query)
    body_embs = np.array([[4, 5, 6], [7, 8, 9], [10, 11, 12]]) # model.encode(body_sentences)
    scores = []
    for emb in body_embs:
        s = util.dot_score(query_emb, emb)[0].cpu().tolist()
        scores.append(s)
    K = max(1,round(k_perc * len(body_sentences)))
    return sorted_K_max(scores, K)

def K_sim_scores(stances_set, body_embs = body_embs, k_perc = 0.25):
    '''
    Given a set of stances and body embeddings, return the mean
    of the K% most semantically similar body sentences
    '''
    N = len(stances_set['Body ID'])
    stances_set = np.array(stances_set)
    scores_pred = []
    for i in range(N):
        query = stances_set[i,0]
        body_id = stances_set[i,1]
        body_emb = body_embs[body_id]
        scores_pred.append(K_sim_scores_single(query, body_emb, k_perc))

    return scores_pred

# Predicting scores on subset of training data
N_small = 1000
stance_train_small = stances[stances['Body ID'] < N_bodies][0:N_small]
scores_pred = K_sim_scores(stance_train_small)

plt.hist(scores_pred)
plt.show()




