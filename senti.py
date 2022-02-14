import pandas as pd
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
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split


import nltk
import re
from nltk.corpus import stopwords


df=pd.read_csv("amazon_baby.csv")
print("df.columns")
 
df['sentiments'] = df.rating.apply(lambda x: 0 if x in [1, 2] else 1)
 
print("top samples in data sets",df.head())
df=shuffle(df)
print("top samples in data sets after shuffling",df.head())
 
#x=df["review"]
#y=df["sentiments"]

X_train,test = train_test_split(df, test_size=0.20, random_state=42)

print("length of training data",len(X_train))
print("length of test data",len(test))


new_train = []
for x in X_train["review"]:
    new_train.append(x)
    #print(new_train)


ultra_train = list(map(lambda ele : str(ele),new_train))


new_test = []
for x in test["review"]:
    new_test.append(x)
    #print(new_test)
    
ultra_test = list(map(lambda ele : str(ele),new_test))


vocab_size = 5000
embedding_dim = 300
max_length = 120
trunc_type = 'post'
oov_tok = '<OOV>'
padding_type = 'post'

tokenizer = Tokenizer(num_words=vocab_size, oov_token=oov_tok)

tokenizer.fit_on_texts(ultra_train)
tokenizer.fit_on_texts(ultra_train)


word_index = tokenizer.word_index
train_seq = tokenizer.texts_to_sequences(ultra_train)
test_seq = tokenizer.texts_to_sequences(ultra_test)

train_paded = pad_sequences(train_seq, maxlen=max_length, truncating=trunc_type)
test_paded = pad_sequences(test_seq, maxlen=max_length, truncating=trunc_type)

size_of_vocabulary=len(tokenizer.word_index) + 1
print("The size of vocabulary ",size_of_vocabulary)


#Create embedding

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
        
model=Sequential()
model.add(Embedding(size_of_vocabulary,300,weights=[embedding_matrix],input_length=570,trainable=False))
model.add(LSTM(128,return_sequences=True,dropout=0.2))
model.add(GlobalMaxPooling1D())
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation='sigmoid'))
model.compile(optimizer='adam', loss='binary_crossentropy',metrics=["acc"])
#es = EarlyStopping(monitor='val_loss', mode='min', verbose=1,patience=3)
#mc=ModelCheckpoint('best_model.h5', monitor='val_acc', mode='max', save_best_only=True,verbose=1)
print(model.summary())


X_train.columns
train_label=X_train["sentiments"]

test_label=test["sentiments"]



model.fit(train_paded,train_label,validation_data=(test_paded,test_label),epochs=2000,batch_size=64)
#Predict on test dataset
prediction=model.predict_classes(test_seq)
Accuracy_Score= accuracy_score(prediction,test_label)
print("Accuracy on test data sets is ", Accuracy_Score)





