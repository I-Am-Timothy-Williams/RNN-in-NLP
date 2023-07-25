# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:15:01 2023

@author: Timot
"""

#Pre-Processing Text Data is important because we need to highlight the attributes that you want your
#machine learning system to pick up on. There are 3 pre-processing steps below::
#1. Remove punctuation
#2. Tokenization
#3. Remove Stopwords



#Read raw data and clean up the column names
import pandas as pd
pd.set_option('display.max_colwidth', 100)

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages = messages.drop(labels =["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages.head()

#Remove Punctuation

#What punctuation is included by default?
import string

string.punctuation

#Why does this matter?
#Periods and parenthesis are just another character to python but it doesnt pull out
#the meaning of a sentence. 
"This message is spam " == "This message is spam."
#Returns False because of the punctuation though the sentence is same

#Define a function to remove punctuation in our messages
def remove_punct(text):
    text = "".join([char for char in text if char not in string.punctuation])
# the "" basically means join at nothing   
    return text

messages ['text_clean'] = messages['text'].apply(lambda x: remove_punct(x))
messages.head()


#Tokenize: Splitting some string or sentence into a list of words

import re

def tokenize (text):
    tokens = re.split('\W+', text)
    return tokens

messages['text_tokenized'] = messages['text_clean'].apply (lambda x: tokenize(x.lower()))
messages.head()

#Remove Stopwords

#Example
tokenize("I am learning NLP".lower()) # lower puts in lowercase

import nltk
stopwords = nltk.corpus.stopwords.words('english')

#Define a function to remove all stopwords

def remove_stopwords(tokenized_text):
    text = [word for word in tokenized_text if word not in stopwords]
    return text

messages['text_nostop'] = messages ['text_tokenized'].apply(lambda x: remove_stopwords(x))
messages.head()


#In example
print(remove_stopwords(tokenize("I am learning NLP".lower())))