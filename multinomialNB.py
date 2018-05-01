# -*- coding: utf-8 -*-
"""
@author: EyadMShokry
"""

from DataPreProcessor import DataPreProcessor
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB
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
                     ('clf', MultinomialNB()),
            ])
##################################################################
#INSTED OF DOING ALL THE ABOVE
#Extracting features from text files
#count_vect = CountVectorizer()
#X_train_counts = count_vect.fit_transform(train_X)
#
#tfidf_transformer = TfidfTransformer()
#X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
#
#
##Building the multinomial Naive Bayes classifier
#model = MultinomialNB()
#model.fit(X_train_tfidf, train_Y)
##################################################################

nb_model = nb_model.fit(train_X, train_Y)

#Prediction
predection = nb_model.predict(test_X)
score = nb_model.score(test_X, test_Y)
print("Multinomial Naive Bayes got an Accuracy of %.2f% %: " % (score * 100))
