from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression as LR, LogisticRegression


def train_Logistic_Regression(x_train,y_train):
    clf = Pipeline([('tfidf', TfidfVectorizer()), ('clf', LogisticRegression()), ])
    clf.fit(x_train, y_train)
    return clf

def Accuracy_Score(x_test,y_test,model):
    predicted= model.predict(x_test)
    return accuracy_score(predicted,y_test)
