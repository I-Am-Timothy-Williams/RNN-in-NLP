# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 09:42:29 2023

@author: Timot
"""
# Intalling/Updating Natural Language ToolKit Library

# //pip install -U nltk

# Import package and download data

# //import nltk
# //nltk.download()

#What's in the package (methods and attributes) ?

# //dir(nltk)

from nltk.corpus import stopwords # stopwords are words that are very common but dont contribute much to the sentence

stopwords.words('english')[0:5] #print first 5 terms for stopwords

stopwords.words ('english')[0:500:25] # print first 500 terms skipping in intervals of 25


