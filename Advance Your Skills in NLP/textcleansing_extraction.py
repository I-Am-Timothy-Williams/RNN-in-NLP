# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 23:22:35 2023

@author: Timot
"""

import os
import nltk

nltk.download('punkt')


with open(os.getcwd()+"/Spark-Course-Description.txt", 'r') as fh:
    filedata = fh.read()

print("Data read from file", filedata[0:200])