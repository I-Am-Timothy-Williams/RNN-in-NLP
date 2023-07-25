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

tagged_docs_tr = [gensim.models.doc2vec.TaggedDocument(v, [i]) for i, v in enumerate(X_train)]

d2v_model = gensim.models.Doc2Vec(tagged_docs_tr,
                                  vector_size = 50,
                                  window = 2,
                                  min_count = 2)

#What does a document vector look like?'

d2v_model.infer_vector(['convert', 'words', 'to', 'vectors'])

#How do we prepare these vectors to be used in a machine learning model?

vector = [[d2v_model.infer_vector(words)] for words in X_test]
