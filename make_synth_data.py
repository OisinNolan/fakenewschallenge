import nltk
from nltk.parse.corenlp import CoreNLPDependencyParser
from nltk.corpus import wordnet
from nltk.tokenize import word_tokenize
from nltk.tokenize.treebank import TreebankWordDetokenizer
twd = TreebankWordDetokenizer()
from lm_scorer.models.auto import AutoLMScorer
import numpy as np
scorer = AutoLMScorer.from_pretrained("distilgpt2")
import csv
import mlconjug3
conjugator = mlconjug3.Conjugator(language='en')
# Must be running coreNLP server locally, 
# see: https://stanfordnlp.github.io/CoreNLP/corenlp-server.html#getting-started
# see docs: https://www.nltk.org/api/nltk.parse.corenlp.html#nltk.parse.corenlp.CoreNLPDependencyParser
dep_parser = CoreNLPDependencyParser(url='http://localhost:9000')

def conjugate(verb, form):
    ends_in_s = verb[-1] == 's'
    tense = 'indicative past tense' if (form in ['VBD'] or not ends_in_s) else 'indicative present'
    person = '3s' if (form in ['VBZ'] or ends_in_s) else '3p'
    return conjugator.conjugate(verb).conjug_info['indicative'][tense][person]

def find_root(deps):
    deps_lines = deps.split('\n')
    for i, dep in enumerate(deps_lines[:-1]):
        split = dep.split('\t')
        if split[3] == 'ROOT':
            return split[0], i

def find_aux(root_idx, deps):
    deps_lines = deps.split('\n')
    for i, dep in enumerate(deps_lines[:-1]):
        split = dep.split('\t')
        if int(split[2]) == (root_idx + 1) and split[3] == 'aux':
            return i


def find_verb_antonym(verb_token, sent_tokens, root_idx):
    best_antonym = verb_token
    best_score = 0
    conj = nltk.pos_tag(sent_tokens)[root_idx][1]
    for syn in wordnet.synsets(verb_token):
        if syn.pos() == 'v':
            for l in syn.lemmas():
                if l.antonyms():
                    for antonym in l.antonyms():
                        ant_conj = conjugate(antonym.name(), conj)
                        sent_tokens[root_idx] = ant_conj
                        score = scorer.sentence_score(twd.detokenize(sent_tokens))
                        if score > best_score:
                            best_score = score
                            best_antonym = ant_conj
    return best_antonym if best_score > 0 else None

def find_not(deps):
    deps_lines = deps.split('\n')
    for i, dep in enumerate(deps_lines[:-1]):
        split = dep.split('\t')
        if split[0].lower() == 'not' and split[3] == 'neg':
            return i

def attempt_negation(sent):
    parse, = dep_parser.raw_parse(sent)
    deps = parse.to_conll(4)
    root, root_idx = find_root(sent, deps)
    tokens = word_tokenize(sent)

    not_id = find_not(deps)
    if not_id:
        del tokens[not_id]
        return twd.detokenize(tokens)

    aux_idx = find_aux(root_idx, deps)
    if aux_idx:
        tokens.insert(aux_idx + 1, 'not')
        return twd.detokenize(tokens)

    antonym = find_verb_antonym(root, tokens, root_idx)
    if antonym:
        tokens[root_idx] = root
        tokens[root_idx] = antonym
        return twd.detokenize(tokens)
    
    return None

# Column keys
HEAD = 0
BODY_ID = 1
STANCE = 2

'''
Sentence negation algorithm:
1) If the word 'not' is in the text, simply remove it
2) If the root verb has an auxiliary verb, add 'not' after it
3) If the root verb has a set of antonyms, choose the antonym that
   yields the sentence with the highest probability, according to GPT-2

Notes:
- Can behave weirdly with *n't words due to tokenization
- Seems to be most reliable for rules (1) and (2), (3) is somewhat hit-or-miss
'''

# Read training stances
with open('./data/combined_stances_train.csv') as f:
    for i, row in enumerate(csv.reader(f)):
        negated = attempt_negation(row[HEAD])
        if negated:
            print(f'\n{i}: \nbefore: {row[HEAD]}\nafter: {negated} \n')