# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 16:37:38 2023

@author: Timot
"""

#Read in Cleaned Text

#Load the cleaned training and test sets

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


X_train = pd.read_csv('./data/X_train.csv')
X_test = pd.read_csv('./data/X_test.csv')
y_train = pd.read_csv('./data/y_train.csv')
y_test = pd.read_csv('./data/y_test.csv')

X_train.head()

#Create TF-IDF Vectors

#Instantiate and fit a TFIDF Vectorizer and then use that trained
#vectorier to transform the messages in the training and test sets

tfidf_vect = TfidfVectorizer()
tfidf_vect.fit(X_train['clean_text'])
X_train_vect = tfidf_vect.transform(X_train['clean_text'])
X_test_vect = tfidf_vect.transform(X_test['clean_text'])

#What words did the vectorizer learn?

tfidf_vect.vocabulary_ #words learned as well as their index

#How are these vectors stored?

#As a sparse matrix, only the nonzero entries along with their location in the matrix

X_test_vect[0]

#Can we convert the vectors to arrays

X_test_vect[0].toarray()

#Fit a basic Random Forest model on these vectors

from sklearn.ensemble import RandomForestClassifier

rf = RandomForestClassifier()
rf_model = rf.fit(X_train_vect,y_train.values.ravel()) #converts to an array so scikit learn is happy

#Use the trained model to make predictions on the test data
y_pred = rf_model.predict(X_test_vect)

#Evaluate the predictions of the model on the holdout test set

from sklearn.metrics import precision_score, recall_score

precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
print('Precision: {} / Recall: {} / Accuracy: {}'.format(
    round(precision, 3), round(recall, 3), round((y_pred==y_test['label']).sum()/len(y_pred), 3)))