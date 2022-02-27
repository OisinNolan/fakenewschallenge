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

from dataset import FakeNewsDataset
from util import *
from baseline import NeuralNetwork

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
import re
alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

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

def K_sim_scores(stances_set, body_embs = body_embs, k_perc = 0.25):
    N = len(stances_set['Body ID'])
    stances_set = np.array(stances_set)
    scores_pred = []
    for i in range(N):
        query = stances_set[i,0]
        query_emb = model.encode(query)
        body_id = stances_set[i,1]
        body_emb = body_embs[body_id]
        scores = util.dot_score(query_emb, body_emb)[0].cpu().tolist()
        K = max(1,round(k_perc * len(docs)))
        scores_pred.append(mean_K_max(scores, K))

    return scores_pred

# Predicting scores on subset of training data
N_small = 1000
stance_train_small = stances[stances['Body ID'] < N_bodies][0:N_small]
scores_pred = K_sim_scores(stance_train_small)

plt.hist(scores_pred)
plt.show()




