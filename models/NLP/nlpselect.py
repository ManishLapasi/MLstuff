from concurrent.futures import process
from matplotlib.pyplot import cla
import numpy as np, pandas as pd
import re

from scipy import rand

dataset = pd.read_csv("../../resources/Part 7 - Natural Language Processing/Section 36 - Natural Language Processing/Python/Restaurant_Reviews.tsv",delimiter="\t", quoting=3)

## cleaning text

def cleanseText(sentence, stopwordSet, porterStemmerFunction):
    processedText = re.sub('[^a-zA-Z]',' ',sentence)        ## replace non letter characters with space
    processedText = (processedText.lower()).split()         ## convert sentence to list of words to process next few steps easily
    processedText = [porterStemmerFunction.stem(word) for word in processedText if not word in stopwordSet]      ## if not a stop-word, then stem the word
    processedText = ' '.join(processedText)                 ## convert list back to sentence
    return processedText

def bayesClassifier(x_train, y_train):
    from sklearn.naive_bayes import GaussianNB
    classifier = GaussianNB()
    classifier.fit(x_train, y_train)
    return classifier

def logisticRegression(x_train, y_train):
    from sklearn.linear_model import LogisticRegression
    classifier = LogisticRegression(random_state=0)
    classifier.fit(x_train,y_train)
    return classifier

def dt(x_train, y_train):
    from sklearn.tree import DecisionTreeClassifier
    classifier = DecisionTreeClassifier(random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def svm(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC(kernel='linear',random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def nonlinearsvm(x_train, y_train):
    from sklearn.svm import SVC
    classifier = SVC(kernel='rbf',random_state=0)
    classifier.fit(x_train, y_train)
    return classifier

def knn(x_train, y_train):
    from sklearn.neighbors import KNeighborsClassifier
    classifier = KNeighborsClassifier(n_neighbors=5, metric='minkowski', p=2)
    classifier.fit(x_train, y_train)
    return classifier

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

cmlist = []
accuracylist = []
precisionlist = []
recalllist = []
f1scorelist = []
from sklearn.metrics import confusion_matrix, accuracy_score

for classifyingFunction in [bayesClassifier, logisticRegression, dt, svm, nonlinearsvm, knn]:
    classifier = classifyingFunction(x_train, y_train)
    y_pred = classifier.predict(x_test)
    #y_list = np.concatenate((y_pred.reshape(len(y_pred),1),y_test.reshape(len(y_test),1)),axis=1)
    #print(y_list)
    cm = confusion_matrix(y_test, y_pred)
    cmlist.append(cm)
    accuracy = accuracy_score(y_test, y_pred)
    accuracylist.append(accuracy)
    precision = cm[0][0] / (cm[0][0] + cm[1][0])
    precisionlist.append(precision)
    recall = cm[0][0] / (cm[0][0] + cm[0][1])
    recalllist.append(recall)
    f1score = 2*precision*recall / (precision + recall)
    f1scorelist.append(f1score)


for i in range(0,len(cmlist)):
    print(cmlist[i])
    print(accuracylist[i],precisionlist[i],recalllist[i],f1scorelist[i])