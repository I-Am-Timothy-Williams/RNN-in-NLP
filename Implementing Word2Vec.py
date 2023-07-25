# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 11:08:00 2023

@author: Timot
"""


# Train Our Model
#Read in the data and clean it

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

#Create tagged document objects to prepare to train the model.
#Tagged documents expects a list of words and a tag for each document, then the doc2vec trains on top
#of the tagged documents. The tag is useful if you have distinct groups of documents. This allows you to 
#pass that information to the doc2vec model if youre doing some sort of clustering

tagged_docs = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)]

#look at waht a tagged document looks like

tagged_docs[0]

#Train a basic doc2vec model

d2v_model = gensim.models.Doc2Vec(tagged_docs,
                                  vector_size = 100,
                                  window = 5,
                                  min_count = 2)

#What happens if we pass in a single word like we did for word2vec?

# d2v_model.infer_vector('text') #will give an error
#What happens if we pass in a list of words?

d2v_model.infer_vector(['i', ' am', 'learning', 'nlp'])