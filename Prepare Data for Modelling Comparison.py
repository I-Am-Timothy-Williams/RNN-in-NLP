# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:26:10 2023

@author: Timot
"""

#Read in and clean data
import numpy as np
import pandas as pd
import nltk
import re
import string
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth', 1000)


stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels = ["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages['label'] = np.where(messages['label'] == 'spam', 1, 0)

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+', text)
    text = [word for word in tokens if word not in stopwords]
    return text

messages['clean_text'] = messages['text'].apply(lambda x: clean_text(x))
messages.head()

X_train, X_test, y_train, y_test = train_test_split(messages['clean_text'],
                                                    messages['label'],
                                                    test_size = 0.2)
#What do the first ten messsages in the training set look like?

X_train[:]

#What do the labels look like?

y_train[:]

#Lets save the training and test sets to ensure we are using the same data for each model

X_train.to_csv('./data/X_train.csv', index=False, header = True)
X_test.to_csv('./data/X_test.csv', index=False, header = True)
y_train.to_csv('./data/y_train.csv', index=False, header = True)
y_test.to_csv('./data/y_test.csv', index=False, header = True)