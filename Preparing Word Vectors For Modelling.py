# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:08:00 2023

@author: Timot
"""

import gensim 
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
pd.set_option('display.max_colwidth',100)

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels =["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]

# clean data using the built in cleaner in gensim

messages['text_clean'] = messages['text'].apply(lambda x: gensim.utils.simple_preprocess(x))

#split into train and test sets

X_train, X_test, y_train, y_test = train_test_split(messages['text_clean'],
                                                    messages['label'], test_size = 0.2)

#Train the word2vec model

w2v_model = gensim.models.Word2Vec(X_train,
                                   vector_size =  100,
                                   window = 5,
                                   min_count = 2)

#Prep Word Vectors

#Generate a list of words the word 2 vec model learned word vectors for
# i.e. all of the words that appeared in the training data at least twice

w2v_model.wv.index_to_key

#generate aggregated sentence vectors based on the word vectors for each word in the sentence

w2v_vect = np.array([np.array([w2v_model.wv[i] for i in ls if i in w2v_model.wv.index_to_key],dtype=object) for ls in X_test],dtype=object)
# Why is the length of the sentence different than the length of the sentence vector?

#for i, v in enumerate:(w2v_vect),:
 #   print(len:X_test.iloc:[i],len(v))
    #iloc finds the location of the text message using the index
    

# Compute Sentence vectors by averaging the word vectors for the words contained in the sentence

w2v_vect_avg = []

for vect in w2v_vect:
    if len(vect) !=0:
        w2v_vect_avg.append(vect.mean(axis=0))
    else:
        w2v_vect_avg.append(np.zeros(100))