# -*- coding: utf-8 -*-
"""
Created on Mon Apr 30 21:12:49 2018

@author: EyadMShokry
"""
from DataPreProcessor import DataPreProcessor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import Pipeline
import time

#Reading Data
DATA_PATH = "F:\Computer Science\Fake-News-Detection\PreProcessedData.csv"
dataProcessor = DataPreProcessor(DATA_PATH)     #You should adjust this path
df = dataProcessor.LoadData()

#Spliting data to training data (80%) & testing data (20%)
train_X, test_X, train_Y, test_Y = train_test_split(df['content'], df['label'], test_size=0.2, random_state=int(time.time()))

text_clf_svm = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf-svm', SGDClassifier(loss='hinge', penalty='l2',                                            
                            alpha=1e-5, n_iter=5, random_state=int(time.time()))),])

_ = text_clf_svm.fit(train_X, train_Y)
predection = text_clf_svm.predict(test_X)
score = text_clf_svm.score(test_X, test_Y)
print(score)