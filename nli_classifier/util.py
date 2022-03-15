from nltk.tokenize import sent_tokenize

MAX_LEN = 50

def pad_truncate(X, K=MAX_LEN):
    '''
    X will be padded with '' or truncated so that it is of length K
    X: array to be padded or truncated
    K: the length you want the output array to be
    '''
    if len(X) > K:
        X = X[:K]
    padding = ['[PAD]' for _ in range(max(0, K - len(X)))]
    return X + padding

def pad_tokenize(X):
    '''
    Preprocess the bodies
    '''
    X_tok = list(map(sent_tokenize, X))
    return list(map(pad_truncate, X_tok))