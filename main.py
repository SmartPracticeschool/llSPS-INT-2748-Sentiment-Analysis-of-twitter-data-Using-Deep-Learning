# -*- coding: utf-8 -*-
"""
Created on Wed Jun 10 10:00:32 2020

@author: Abhishek Upadhyay
@author: Deepak Kumar
"""

import numpy as np
import pandas as pd
import re
import nltk

nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

df = pd.read_csv("training.1600000.processed.noemoticon1.csv")
df_column = [['emotion', 'tweet']]
classes = df[df_column[0]].size

ps = PorterStemmer()
data = []
for i in range(0, 10000):
    tweet = df["tweet"][i]
    tweet = re.sub('[^a-zA-z]', ' ', tweet)
    tweet = tweet.lower()
    tweet = tweet.split()
    tweet = [ps.stem(word) for word in tweet if not word in set(stopwords.words('english'))]
    tweet = ' '.join(tweet)
    data.append(tweet)
print(tweet)

x = df[['emotion', 'tweet']]

y = df['emotion'].values

from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2)

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.layers import Dense
from keras.models import Sequential
from keras.models import load_model

"""To label the sentiment classes using integers. Not to be used for the neural network"""


def to_categorical(df):
    df.sentiment = pd.Categorical(df.sentiment)
    df['class'] = df.sentiment.cat.codes
    return df['class']


"""Function returns the one-hot representation of the sentiment classes"""


def to_OneHot(df, df_columns):
    b = pd.get_dummies(df[df_column[0]], prefix="")
    list1 = list(b)
    OneHot = b[list1[0]]
    OneHot = np.column_stack(b[list1[i]] for i in range(len(list1)))
    print(len(list1))
    print(OneHot)
    return OneHot


"""Labels can be either to_OneHot function return value or to_categorical function return value"""


def split_train_test(df, df_column, labels, test_split=0.2):
    x_train, x_test, y_train, y_test = train_test_split(df[df_column[1]], labels, test_size=test_split, random_state=10)
    return x_train, x_test, y_train, y_test

def create_model():
  model = tf.keras.models.Sequential([
    keras.layers.Dense(512, activation='relu', input_shape=(784,)),
    keras.layers.Dropout(0.2),
    keras.layers.Dense(10)
  ])

  model.compile(optimizer='adam',
                loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=['accuracy'])

  return model


def model_train(x_train, y_train, batch_size=15, epochs=15, optimizer='adam', loss='categorical_crossentropy',
                activation='tanh', hidden_neurons=30):
    model = Sequential()
    model.add(Dense(units=784, input_dim=x_train.shape[1], activation=activation))
    model.add(Dense(hidden_neurons, activation=activation))
    model.add(Dense(classes, activation='softmax'))
    model.compile(optimizer=optimizer, loss=loss, metrics=['accuracy'])
    model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    return model


def eval_model(x_test, y_test, model):
    scores = model.evaluate(x_test, y_test)
    print("\n%s: %.2f%%" % (model.metrics_names[1], scores[1] * 100))
model= create_model()
model.summary()