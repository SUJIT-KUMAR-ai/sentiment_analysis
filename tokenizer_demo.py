import pandas as pd
import numpy as np
from tensorflow.keras.models import *
from tensorflow.keras.layers import *
from tensorflow.keras.layers import Embedding
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.preprocessing.text import one_hot
from tensorflow.keras.layers import LSTM
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Bidirectional
from tensorflow.keras.layers import Dropout
from sklearn.metrics import accuracy_score


import nltk
import re
from nltk.corpus import stopwords

tokenizer = Tokenizer()

text=["The ruling BJP announced the names of candidates for nine seats that will go to polls in the seventh and final phase of the ongoing Uttar Pradesh Assembly election"]

tokenizer.fit_on_texts(text)

word_index=tokenizer.word_index

print("number of word in vocabulary",len(word_index))



print("words in vocab",word_index)

text_sequence=tokenizer.texts_to_sequences(text)

print("word in sentences are replaced with word ID",text_sequence)


size_of_vocabulary=len(tokenizer.word_index) + 1
print("The size of vocabulary ",size_of_vocabulary)

embeddings_index = dict()


f = open('glove.6B.300d.txt')

for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    embeddings_index[word] = coefs

f.close()
print('Loaded %s word vectors.' % len(embeddings_index))


embedding_matrix = np.zeros((size_of_vocabulary, 300))

for word, i in tokenizer.word_index.items():
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        embedding_matrix[i] = embedding_vector







