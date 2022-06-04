from concurrent.futures import process
from matplotlib.pyplot import cla
import numpy as np, pandas as pd
import re

dataset = pd.read_csv("../../resources/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python/Restaurant_Reviews.tsv",delimiter="\t", quoting=3)

## cleaning text

def cleanseText(sentence, stopwordSet, porterStemmerFunction):
    processedText = re.sub('[^a-zA-Z]',' ',sentence)        ## replace non letter characters with space
    processedText = (processedText.lower()).split()         ## convert sentence to list of words to process next few steps easily
    processedText = [porterStemmerFunction.stem(word) for word in processedText if not word in stopwordSet]      ## if not a stop-word, then stem the word
    processedText = ' '.join(processedText)                 ## convert list back to sentence
    return processedText

import nltk
nltk.download('stopwords')          ## stopwords are not useful for computation (like 'the', 'a', 'an')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []
num_reviews = 1000

stopwords_list = set(stopwords.words('english'))
stopwords_list.remove('not')
stopwords_set = set(stopwords_list)
ps = PorterStemmer()            ## import word stemmer -> this bunches up tenses of the same word (for eg. like and liked are the same word)
    
for uncleanedReview in range(0,num_reviews):
    review = cleanseText(dataset['Review'][uncleanedReview],stopwords_set,ps)
    corpus.append(review)           ## append the cleaned sentence to the corpus

## Bag of words model

from sklearn.feature_extraction.text import CountVectorizer
cv = CountVectorizer(max_features = 1500)
x = cv.fit_transform(corpus).toarray()
y = dataset.iloc[:,-1].values

## split into train and test set

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2,random_state=0)

from sklearn.naive_bayes import GaussianNB
classifier = GaussianNB()
classifier.fit(x_train, y_train)
y_pred = classifier.predict(x_test)

y_list = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
print(y_list)

from sklearn.metrics import confusion_matrix, accuracy_score
cm = confusion_matrix(y_test, y_pred)
print(cm)
print(accuracy_score(y_test, y_pred))

reviewList = ['is good place','is bad place']
for review in reviewList:
    cleansedReview = cleanseText(review,stopwords_set,ps)
    print(cleansedReview)
    reviewVector = cv.transform([cleansedReview]).toarray()
    print(reviewVector)
    print(classifier.predict(reviewVector))