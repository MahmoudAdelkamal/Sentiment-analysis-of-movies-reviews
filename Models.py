from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR, LogisticRegression
from sklearn.svm import SVC


def train_Logistic_Regression(x_train,y_train):
    clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression()), ])
    clf.fit(x_train, y_train)
    return clf

def train_svc(x_train,y_train) :
    clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', SVC()), ])
    clf.fit(x_train, y_train)
    return clf


