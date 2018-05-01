# -*- coding: utf-8 -*-
"""
@author: EyadMShokry
"""

from DataPreProcessor import DataPreProcessor
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
import time
import matplotlib.pyplot as plt

#Reading Data
dataProcessor = DataPreProcessor("F:\Computer Science\Fake-News-Detection\PreProcessedData.csv")
df = dataProcessor.LoadData()

#Spliting data to training data (80%) & testing data (20%)
train_X, test_X, train_Y, test_Y = train_test_split(df['content'], df['label'], test_size=0.2, random_state=int(time.time()))

#Buliding Pipeline
kVals = []
accuracies = []
for k in range(1,11):
    text_clf = Pipeline([('vect', CountVectorizer()),
                         ('tfidf', TfidfTransformer()),
                         ('clf', KNeighborsClassifier(n_neighbors=k)),])
    text_clf = text_clf.fit(train_X, train_Y)
    #Prediction
    predection = text_clf.predict(test_X)
    score = text_clf.score(test_X, test_Y)
    kVals.append(k)    
    accuracies.append(score)
    print("k=%d, accuracy=%.2f%%" % (k, score * 100))

plt.title("Model's Accuracy comparing to K values")    
plt.plot(kVals, accuracies)
plt.xlabel("K Values")
plt.ylabel("Accuracy")
plt.show()
