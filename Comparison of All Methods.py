# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 18:27:17 2023

@author: Timot
"""

#Lessons to Learn
# Precision is basically the ability of the model to call the Real emails Real.
# Recall is basically the ability of the model to call False emails False
# Accuracy is the ability of the model to get any calls correct

#TFIDF

#Simple Document Vectors that represent how important a word is to a document within a corpus
#No consideration of context in which a word is used
#Creates Very Sparse, Large Vectors
#Quick and easy way to set a baseline

#Word2Vec

#Word Vectors Created Through a Shallow, Two-Layer Neural Network; 
#Word Vectors can then be averaged or summed to create document level vectors
#Creates smaller, dense vectors
#Word vectors do have context built in
#Information is lost when averaging or summing word vectors

#Doc2Vec

#Document level vectors through a shallow two layer neural network
#Creates smaller, dense vectors
#Document vectors do have context built in
#Slower but better than word2vec for sentence vectors

#RNN

#A type of neural network that has an understanding of the datas sequential nature
#Dense vectors are created within the model
#RNN is extremely powerful even with limited data

