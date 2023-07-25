# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 15:19:47 2023

@author: Timot
"""

#What makes RNNs so powerful for NLP Problems?

#TFIDF took words independently. One spot in the vector per word (large vectors)
#Word2Vec and Doc2Vec have small dense vectors that tries to capture context with awindow
#RNN ingests the text in the same that we read. It evaluates each word within the context of the word that came before it,
    #then at the end of the sentence it has a pretty good idea of the message the sentence was trying to convey