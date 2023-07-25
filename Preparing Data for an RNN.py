# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:25:17 2023

@author: Timot
"""

#How to Implement a Basic RNN

#Read In, Clean, And Split The Data

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 1000)

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
labels = np.where(messages['label'] == 'spam', 1, 0)

X_train, X_test, y_train, y_test = train_test_split(messages['text'],
                                                    labels,
                                                    test_size = 0.2)

#Prep Data For Modeling

#Install Keras
# !pip install -U keras
# !pip install -U tensorflow

#Import the tools we will need rom keras
from keras.preprocessing.text import Tokenizer
from keras.utils import pad_sequences

#Initialize and fit the tokenozer 

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_train)

#Use that tokenizer to transform the text messages in the training and test sets

X_train_seq = tokenizer.texts_to_sequences(X_train)
X_test_seq = tokenizer.texts_to_sequences(X_test)

#this converts the text message string into a list of integers where each integer represents the index of the word in the trained tokenizer

#What do these sequences look like?
X_train_seq[0] #integer representation of the first text message in the trained data set

#Pad the sequences so each sequence is the same length
#RNN model requires each sentence or list of integers to be the same length. Word2Vec used averaging to ensure same length, but with RNN
#We need to pad sequences (by adding zeros or truncating values)

X_train_seq_padded = pad_sequences(X_train_seq, 50)
X_test_seq_padded = pad_sequences(X_test_seq, 50)

#What do these padded sequences look like?
X_train_seq_padded[0]


