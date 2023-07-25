# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:03:53 2023

@author: Timot
"""

import gensim
import numpy as np
import pandas as pd

X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv')
y_test = pd.read_csv('./data/y_test.csv')

#Create Tagged Document vectors for each text message in the training and test sets

tagged_docs_train = [gensim.models.doc2vec.TaggedDocument(v,[i])
                     for i, v in enumerate(X_train['clean_text'])]
tagged_docs_test = [gensim.models.doc2vec.TaggedDocument(v,[i])
                     for i, v in enumerate(X_test['clean_text'])]

#What do theese Tagged Document object look like?

tagged_docs_train[:10]

#Train a basic doc2vec model

d2v_model = gensim.models.Doc2Vec(tagged_docs_train,
                                  vector_size = 100,
                                  window = 5,
                                  min_count = 2)

#Infer the vectors to be used in training and testing

train_vectors = [d2v_model.infer_vector(eval(v.words)) for v in tagged_docs_train]
test_vectors = [d2v_model.infer_vector(eval(v.words)) for v in tagged_docs_test]

#Fit RandomForestClassifier on Top of Document Vectors

#Fit a basic model, make predictions and generate evaluation

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, recall_score

rf = RandomForestClassifier()
rf_model = rf.fit(train_vectors, y_train.values.ravel())

y_pred = rf_model.predict(test_vectors)

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test['label']).sum()/len(y_pred), 3)))




