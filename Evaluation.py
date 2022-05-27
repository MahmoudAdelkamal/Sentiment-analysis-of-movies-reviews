from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import *
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

def Accuracy_Score(x_test,y_test,model):
    predicted= model.predict(x_test)
    return accuracy_score(predicted,y_test)

def classificationReport(x_test,y_test,model):
    predicted= model.predict(x_test)
    return classification_report(y_test,predicted)

def ConfusionMatrix(x_test,y_test,model) :
    predicted= model.predict(x_test)
    cf_matrix = confusion_matrix(y_test,predicted)
    return cf_matrix