# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:45:21 2023

@author: Timot
"""

#Read in and Clean Text

import pandas as pd
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer

stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels =["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [word for word in tokens if word not in stopwords]
    return text

tfidf_vect = TfidfVectorizer(analyzer = clean_text)
X_tfidf = tfidf_vect.fit_transform(messages['text'])

X_features = pd.DataFrame(X_tfidf.toarray())


#Explore RandomForestClassifier Attributes and Hyperparameters

from sklearn.ensemble import RandomForestClassifier

print(RandomForestClassifier()) #max depth is how deep your decision tree will be. None means  it will keep going until it minimizes loss
# n_estimators is the number of trees it will be. default is 100

#Import the methods that will be needed to evaluate a basic model

from sklearn.metrics import precision_score, recall_score
from sklearn.model_selection import train_test_split

#Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_features, messages['label'], test_size = 0.2)

#Fit a basic random forest model

rf = RandomForestClassifier()
rf_model = rf.fit(X_train, y_train)

#make prediciotns on the test set using the fit model
y_pred = rf_model.predict(X_test)

#Evaluate model predictions using precision and recall
precision = precision_score(y_test, y_pred, pos_label='spam')
recall = recall_score(y_test, y_pred, pos_label='spam')
print('Precision: {} / Recall: {}' .format(round(precision, 3), round(recall, 3)))