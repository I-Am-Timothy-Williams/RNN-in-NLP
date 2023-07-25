# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:58:59 2023

@author: Timot
"""
# Read in and view the raw data

import pandas as pd

messages = pd.read_csv('./data/spam.csv', encoding = 'latin-1')
messages.head()

# drop unecessary columns and label columns that will be used

messages = messages.drop(labels =["Unnamed: 2", "Unnamed: 3", "Unnamed: 4"], axis = 1)
messages.columns = ["label", "text"]
messages.head()

#How big is the dataset?

messages.shape

#What portion of our text messages are actually spam?

messages['label'].value_counts()
#If you get an imbalance with ratio of 50:1 or 100:1 you may want to counter the imbalance
#by downsampling the majority class, altering loss function to penalize one class more, or upsampling
#the majority class

#Are we missing any data?
print('Number of nulls in label: {}'.format(messages['label'].isnull().sum()))
print('Number of nulls in text: {}'.format(messages['text'].isnull().sum()))