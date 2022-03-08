import pandas as pd
import numpy as np
from util import *
from sentence_transformers import SentenceTransformer, util
from datetime import datetime
import pickle
from dataset import FakeNewsDataset
from torch.utils.data import DataLoader

BATCH_SIZE=10

# Pre-processing
stances = pd.read_csv('combined_stances_train.csv')
bodies = pd.read_csv('combined_bodies_train.csv')
vectorizer = TfidfVectorizer()
vectorizer.fit(np.concatenate((stances['Headline'].values, bodies['articleBody'].values), axis=None))

# load dataset
training_data = FakeNewsDataset('train_stances.csv', 'train_bodies.csv')
train_dataloader = DataLoader(training_data, batch_size=BATCH_SIZE)

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

#N_bodies = 10
N_bodies = len(bodies['Body ID'])
print(f'Extracting {N_bodies} body embeddings')

def embed_bodies(bodies_set):
    body_embeds = {}
    for body_ID in range(N_bodies):
    #for body_ID in range(n_bodies):
        sen_splits = split_into_sentences(bodies_set['articleBody'][body_ID])
        if len(sen_splits) == 0:
            sen_splits = bodies_set['articleBody'][body_ID]
        body_embeds[bodies_set['Body ID'][body_ID]] = model.encode(sen_splits)
    return body_embeds

t0 = datetime.now()
body_embs = embed_bodies(bodies)
t1 = datetime.now()
print(f'{N_bodies} bodies embedded in ',t1-t0)

file_name = f'body_embs_{N_bodies}.pkl'

open_file = open(file_name, "wb")
pickle.dump(body_embs, open_file)
open_file.close()

print('Embeddings saved succesfully')