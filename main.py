# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:00:32 2020

@author: Abhishek Upadhyay
"""

import numpy as np
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

dataset = pd.read_csv("training.1600000.processed.noemoticon1.csv")

ps = PorterStemmer()
data = []
for i in range(0, 10000):
    tweet = dataset["tweet"][i]
    tweet = re.sub('[^a-zA-z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    data.append(tweet)
print(tweet)

x = dataset[['emotion', 'tweet']]

y = dataset['emotion']

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import keras
