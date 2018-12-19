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
import matplotlib
from matplotlib import pyplot
%matplotlib inline
# import ptvsd

# ptvsd.enable_attach(address=('0.0.0.0','5678'))
# ptvsd.wait_for_attach()

def load_and_prec():
    train_df = pd.read_csv("../input/train.csv")
    test_df = pd.read_csv("../input/test.csv")
    print("Train shape : ",train_df.shape)
    print("Test shape : ",test_df.shape)
    
    ## fill up the missing values
    train_X = replace_special_char(train_df).fillna("_##_").values
    test_X = replace_special_char(test_df).fillna("_##_").values

    ## Tokenize the sentences
    tokenizer = Tokenizer(num_words=max_features, filters='`!"#$%&()*+,-/:;<=>@[\]^_{|}~')
    # tokenizer = Tokenizer(num_words=max_features, filters='`!"#%&*+/;<=>@[\]^_{|}~')
    tokenizer.fit_on_texts(list(train_X))
    train_X = tokenizer.texts_to_sequences(train_X)
    print('max length is ',pd.DataFrame(train_X).values.shape)
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
    return df["question_text"].apply(lambda s: s.replace("?", " ?")).apply(lambda s: s.replace(".", " ."))
    # return df["question_text"].apply(lambda s: s.replace("?", " ?")).apply(lambda s: s.replace(".", " .")).apply(lambda s: s.replace("(", " ( ")).apply(lambda s: s.replace(")", " ) ")).apply(lambda s: s.replace(":", " : ")).apply(lambda s: s.replace("-", " - ")).apply(lambda s: s.replace(",", " ,")).apply(lambda s: s.replace("$", " $ "))
    
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
    
def model_lstm_atten(embedding_matrix):
    
    inp = Input(shape=(maxlen,))
    x = Embedding(max_features, embed_size, weights=[embedding_matrix], trainable=False)(inp)
    x = SpatialDropout1D(0.5)(x)
    x = Bidirectional(CuDNNLSTM(100, return_sequences=True))(x)
    x = Attention(maxlen)(x)
    # x = GlobalMaxPooling1D()(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.5)(x)
    outp = Dense(1, activation="sigmoid")(x)    

    model = Model(inputs=inp, outputs=outp)
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    return model

# Train
def train(model, train_X, train_y, val_X, val_y, callback=None):
    history = model.fit(train_X, train_y, batch_size=512, epochs=5, validation_data=(val_X, val_y), callbacks = callback, verbose=0)
    return history

train_X, test_X, train_y, word_index = load_and_prec()
embedding_matrix = load_glove(word_index)
np.shape(embedding_matrix)

class Attention(Layer):
    def __init__(self, step_dim,
                 W_regularizer=None, b_regularizer=None,
                 W_constraint=None, b_constraint=None,
                 bias=True, **kwargs):
        self.supports_masking = True
        self.init = initializers.get('glorot_uniform')

        self.W_regularizer = regularizers.get(W_regularizer)
        self.b_regularizer = regularizers.get(b_regularizer)

        self.W_constraint = constraints.get(W_constraint)
        self.b_constraint = constraints.get(b_constraint)

        self.bias = bias
        self.step_dim = step_dim
        self.features_dim = 0
        super(Attention, self).__init__(**kwargs)

    def build(self, input_shape):
        assert len(input_shape) == 3

        self.W = self.add_weight((input_shape[-1],),
                                 initializer=self.init,
                                 name='{}_W'.format(self.name),
                                 regularizer=self.W_regularizer,
                                 constraint=self.W_constraint)
        self.features_dim = input_shape[-1]

        if self.bias:
            self.b = self.add_weight((input_shape[1],),
                                     initializer='zero',
                                     name='{}_b'.format(self.name),
                                     regularizer=self.b_regularizer,
                                     constraint=self.b_constraint)
        else:
            self.b = None

        self.built = True

    def compute_mask(self, input, input_mask=None):
        return None

    def call(self, x, mask=None):
        features_dim = self.features_dim
        step_dim = self.step_dim

        eij = K.reshape(K.dot(K.reshape(x, (-1, features_dim)),
                        K.reshape(self.W, (features_dim, 1))), (-1, step_dim))

        if self.bias:
            eij += self.b

        eij = K.tanh(eij)

        a = K.exp(eij)

        if mask is not None:
            a *= K.cast(mask, K.floatx())

        a /= K.cast(K.sum(a, axis=1, keepdims=True) + K.epsilon(), K.floatx())

        a = K.expand_dims(a)
        weighted_input = x * a
        return K.sum(weighted_input, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape[0],  self.features_dim

DATA_SPLIT_SEED = 2018

splits = list(StratifiedKFold(n_splits=3, shuffle=True, random_state=DATA_SPLIT_SEED).split(train_X, train_y))
for idx, (train_idx, valid_idx) in enumerate(splits):
    X_train = train_X[train_idx]
    y_train = train_y[train_idx]
    X_val = train_X[valid_idx]
    y_val = train_y[valid_idx]
    model = model_lstm_atten(embedding_matrix)
    trn = pd.DataFrame()
    val = pd.DataFrame()
    n_repeats = 7
    for e in range(n_repeats):
        history = train(model, X_train, y_train, X_val, y_val)
        trn[str(e)] = history.history['loss']
        val[str(e)] = history.history['val_loss']

    for i in trn.columns:
        pyplot.plot(trn[i], color=(0,0,1 - (int(i)+1)/(n_repeats+2)), label=i)
        pyplot.plot(val[i], color=(1 - (int(i)+1)/(n_repeats+2),0,0), label=i)
    pyplot.title('model train vs validation loss')
    pyplot.ylabel('loss')
    pyplot.xlabel('epoch')
    pyplot.show()   
    

