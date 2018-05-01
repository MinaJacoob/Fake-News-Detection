# -*- coding: utf-8 -*-
"""
Created on Tue May  1 02:50:34 2018

@author: EyadMShokry
"""

from DataPreProcessor import DataPreProcessor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
import time

#Reading Data
dataProcessor = DataPreProcessor("F:\Computer Science\Fake-News-Detection\PreProcessedData.csv")
df = dataProcessor.LoadData()

#Spliting data to training data (80%) & testing data (20%)
train_X, test_X, train_Y, test_Y = train_test_split(df['content'], df['label'], test_size=0.2, random_state=int(time.time()))

#Buliding Pipeline
nb_model = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', LinearSVC()),
            ])

nb_model = nb_model.fit(train_X, train_Y)

#Prediction
predection = nb_model.predict(test_X)
score = nb_model.score(test_X, test_Y)
print("Linear Support Vector Classification got an Accuracy of %.2f% %: " % (score * 100))
