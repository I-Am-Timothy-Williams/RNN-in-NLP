# -*- coding: utf-8 -*-
"""
Created on Thu Jun 29 10:56:16 2023

@author: Timot
"""

#word2vec
# is a shallow, two-layer neural network that accepts a text corupus as an input, and
# it returns a set of vectors (aka embeddings); each vector is a numeric representation of a word
# a skip grab is basicaly where the model looks at each word one by one and looks at the words within a window of words, before and after.
# with this it understands the context of the word.