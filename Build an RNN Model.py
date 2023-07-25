# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:12:28 2023

@author: Timot
"""

from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences
import pandas as pd

X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv')
y_test = pd.read_csv('./data/y_test.csv')

#Train the tokenizer and use that tokenizer to convert the sentences to sequences of numbers
#Where each number represents the index of the words stored in the tokenizer

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train['clean_text'])
X_train_seq = tokenizer.texts_to_sequences(X_train['clean_text'])
X_test_seq = tokenizer.texts_to_sequences(X_test['clean_text'])

#Pad the sequences so each sequence is the same length
X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)

#Build and Evaluate RNN

import keras.backend as K
from keras.layers import Dense, Embedding, LSTM
from keras.models import Sequential

def recall_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true *y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def precision_m(y_true, y_pred):
    true_positives = K.sum(K.round(K.clip(y_true *y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision

# Construct a simple RNN model

model = Sequential()
model.add(Embedding(len(tokenizer.index_word)+1, 32))
model.add(LSTM(32, dropout = 0, recurrent_dropout = 0)) #LSTM is like an RNN as well, Long Short Term Memory
model.add(Dense(32, activation = 'relu'))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

#Compile the model

model.compile(optimizer = 'adam',
              loss = 'binary_crossentropy',
              metrics = ['accuracy', precision_m, recall_m])

# Fit the RNN

history = model.fit(X_train_seq_padded, y_train['label'],
                     batch_size = 32, epochs = 10,
                     validation_data = (X_test_seq_padded, y_test))

#Plot basic evaluation metrics across epochs

import matplotlib.pyplot as plt 
#%matplotlib inline

for i in ['accuracy', 'precision_m', 'recall_m']:
    acc = history.history[i]
    val_acc = history.history['val_{}'.format(i)]
    epochs = range(1, len(acc) +1)
    
    plt.figure()
    plt.plot(epochs, acc, label = 'Training Accuracy')
    plt.plot(epochs, val_acc, label = 'Validation Accuracy')
    plt.title('Results for {}'.format(i))
    plt.legend()
    plt.show()
