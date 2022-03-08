from re import S
from torch.utils.data import DataLoader
import torch
from torch import nn
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd
import numpy as np

import warnings
warnings.filterwarnings('ignore')

from dataset import FakeNewsDataset
from util import *
from baseline import NeuralNetwork

from sentence_transformers import SentenceTransformer, util

import matplotlib.pyplot as plt
from datetime import datetime
import pickle

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
def stance_to_rel_un(stan):
    rel_un = np.array(stan)
    for i in range(len(rel_un)):
        if rel_un[i] != 'unrelated':
            rel_un[i] = 'related'
    return rel_un

def stance_to_bin(stan):
    stan = np.array(stan)
    rel_un = np.zeros(len(stan))
    for i in range(len(rel_un)):
        if stan[i] != 'unrelated':
            rel_un[i] = 1
    return rel_un

rel_un_train = stance_to_bin(stances['Stance'])

# Load Embeddings
N_bodies = 6131

file_name = f'body_embs_{N_bodies}.pkl'
open_file = open(file_name, "rb")
body_embs = pickle.load(open_file)
open_file.close()

# Get scores from HuggingFace Sentence-Bert
model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-cos-v1')

def mean_K_max(scores,K):
    scores.sort(reverse=True)
    k_largest = scores[0:K]
    return np.mean(k_largest)

def K_sim_scores(stances_set, body_embs = body_embs, k_perc = 0.50):
    N = len(stances_set['Body ID'])
    stances_set = np.array(stances_set)
    scores_pred = []
    for i in range(N):
        query = stances_set[i,0]
        query_emb = model.encode(query)
        body_id = stances_set[i,1]
        body_emb = body_embs[body_id]
        scores = util.dot_score(query_emb, body_emb)[0].cpu().tolist()
        K = max(1,round(k_perc * len(scores)))
        scores_pred.append(mean_K_max(scores, K))

    return scores_pred

# Predicting scores on subset of training data
N_train = 2000
print(f'Predicting related-unrelated for {N_train} heads')
stance_train_small = stances[:N_train]
scores_pred = K_sim_scores(stance_train_small)

# Compare to actual rel-un
rel_un_small = stance_to_bin(stance_train_small['Stance'])

scores_un = np.array(scores_pred)[rel_un_small < 0.5]
scores_rel = np.array(scores_pred)[rel_un_small > 0.5]


fig, ax = plt.subplots()
ax.boxplot([scores_un, scores_rel])
ax.set_xticklabels(['Unrelated','Related'])
ax.set_ylabel('Mean similarity score of 25% most similar sentences')
fig.show()



########## Analysis of Similarity Scores ##########

def return_sim_scores(stances_set, body_embs = body_embs):
    N = len(stances_set['Body ID'])
    stances_set = np.array(stances_set)
    sim_scores = []
    #sim_scores = np.zeros()
    for i in range(N):
        query = stances_set[i,0]
        query_emb = model.encode(query)
        body_id = stances_set[i,1]
        body_emb = body_embs[body_id]
        scores = util.dot_score(query_emb, body_emb)[0].cpu().tolist()
        sim_scores.append(scores)

    return sim_scores

N_train = 2000
stance_train_small = stances[:N_train]
sim_scores = return_sim_scores(stance_train_small)

# Histogram of number of embeddings (sentences) in body
num_embeds = np.zeros(N_train)
for i in range(N_train):
    num_embeds[i] = len(sim_scores[i])

fig, ax = plt.subplots()
ax.hist(num_embeds, bins = 50)
ax.set_xlabel('Number of sentences in body')
fig.show()

# Bodies with only 1 embedding
single_ladies = []
for i in range(N_train):
    if num_embeds[i] == 1:
        single_ladies.append(i)

def head_to_body(head_index):
    body_id = stances['Body ID'][head_index]
    return bodies['articleBody'][int(np.where(bodies['Body ID'] == body_id)[0])]

for i in range(10):
    print(head_to_body(single_ladies[i]))
# SOMEHOW A LOT OF THIS 1-SENTENCE BODIES ARE ABOUT ISIS?

import pandas as pd
fig, ax = plt.subplots()
df = pd.DataFrame({'num_embeds':num_embeds, 'scores':scores_pred,'rel_un':stance_to_rel_un(stance_train_small['Stance'])})
cols = {'unrelated':'r','related':'g'}
ax.scatter(df['num_embeds'], df['scores'], c=df['rel_un'].map(cols), s=5)
ax.set_xlim([0,120])
ax.legend('upper right')
fig.show()


