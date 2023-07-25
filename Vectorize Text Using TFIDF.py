# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:35:34 2023

@author: Timot
"""

#Term Frequency - Inverse Document Frequency
# Creates a document term matrix; one row per document, one column per word in the corpus
#Generates a weighting for each word/document pair intented to reflect how important a given word is to the document
#within the context of its frequency within a larger corpus

#Read in Text

#Read in raw data and clean up the column names
import pandas as pd
import re
import string
import nltk
pd.set_option('display.max_colwidth',100)

stopwords = nltk.corpus.stopwords.words('english')

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels =["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages.head()

#Create function to clean text

def clean_text(text):
    text = "".join([word.lower() for word in text if word not in string.punctuation])
    tokens = re.split('\W+',text)
    text = [word for word in tokens if word not in stopwords]
    return text

#Apply TFidf Vectorizer

#Fit a basic TFIDF Vectorizer and view the results
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vect = TfidfVectorizer(analyzer = clean_text)
X_tfidf = tfidf_vect.fit_transform(messages['text'])
#print(X_tfidf.shape)
#print(tfidf_vect.get_feature_names())


#How is the output of TFidfVectorizer stored?

#Outputs a sparse matrix (most entries are 0). Only stores elements of the 9 zero elements
