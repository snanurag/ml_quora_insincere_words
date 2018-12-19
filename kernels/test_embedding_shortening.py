import os
print(os.listdir("../input"))

# Any results you write to the current directory are saved as output.

## some config values 
embed_size = 300 # how big is each word vector
max_features = 100000 # how many unique words to use (i.e num rows in embedding vector)
maxlen = 60 # max number of words in a question to use

import os
import time
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from tqdm import tqdm
import math
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Input, CuDNNLSTM, Embedding, Dropout, Activation, CuDNNGRU, Conv1D
from keras.layers import Bidirectional, GlobalMaxPool1D, GlobalMaxPooling1D, GlobalAveragePooling1D
from keras.layers import Input, Embedding, Dense, Conv2D, MaxPool2D, concatenate
from keras.layers import Reshape, Flatten, Concatenate, Dropout, SpatialDropout1D
from keras.optimizers import Adam
from keras.models import Model
from keras import backend as K
from keras.engine.topology import Layer
from keras import initializers, regularizers, constraints, optimizers, layers
from keras.layers import concatenate
from keras.callbacks import *
import ptvsd

ptvsd.enable_attach(address=('0.0.0.0','5678'))
ptvsd.wait_for_attach()

def load_and_prec():
    train_df = pd.read_csv("../input/train_test.csv")
    test_df = pd.read_csv("../input/test_test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## fill up the missing values
    train_df.replace('a', 'b', inplace=True)
    train_X = replace_special_char(train_df).fillna("_##_").values
    test_X = replace_special_char(test_df).fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters='`!"#%&*+/;<=>@[\]^_{|}~')
    tokenizer.fit_on_texts(list(train_X))
    tokenizer.fit_on_texts(list(test_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    test_X = tokenizer.texts_to_sequences(test_X)

    ## Pad the sentences 
    train_X = pad_sequences(train_X, maxlen=maxlen)
    test_X = pad_sequences(test_X, maxlen=maxlen)

    ## Get the target values
    train_y = train_df['target'].values
    
    #shuffling the data
    np.random.seed(2018)
    trn_idx = np.random.permutation(len(train_X))

    train_X = train_X[trn_idx]
    train_y = train_y[trn_idx]
    
    return train_X, test_X, train_y, tokenizer.word_index

def replace_special_char(df):
    return df["question_text"].apply(lambda s: s.replace("?", " ?")).apply(lambda s: s.replace(".", " .")).apply(lambda s: s.replace("(", " ( ")).apply(lambda s: s.replace(")", " ) ")).apply(lambda s: s.replace(":", " : ")).apply(lambda s: s.replace("-", " - ")).apply(lambda s: s.replace(",", " ,")).apply(lambda s: s.replace("$", " $ "))

def load_glove(word_index):
    EMBEDDING_FILE = '../input/embeddings/glove.840B.300d/glove.840B.300d.txt'
    def get_coefs(word, *arr): return word, np.asarray(arr, dtype='float32')
    embeddings_index = dict(get_coefs(*o.split(" ")) for o in open(EMBEDDING_FILE))

    all_embs = np.stack(embeddings_index.values())
    emb_mean,emb_std = all_embs.mean(), all_embs.std()
    embed_size = all_embs.shape[1]

    # word_index = tokenizer.word_index
    nb_words = min(max_features, len(word_index))
    embedding_matrix = np.random.normal(emb_mean, emb_std, (nb_words, embed_size))
    for word, i in word_index.items():
        if i >= max_features: continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None: embedding_matrix[i] = embedding_vector
            
    return embedding_matrix 

train_X, test_X, train_y, word_index = load_and_prec()
embedding_matrix = load_glove(word_index)
np.shape(embedding_matrix)
