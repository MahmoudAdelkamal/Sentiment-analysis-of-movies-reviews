from sklearn.metrics import *
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.linear_model import LogisticRegression as LR

def train_Decition_Tree(x_train,y_train):
    model = DT()
    model.fit(x_train, y_train)
    return model

def train_Logistic_Regression(x_train,y_train):
    model = LR()
    model.fit(x_train, y_train)
    return model

def Accuracy_Score(x_test,y_test,model):
    predicted= model.predict(x_test)
    return accuracy_core(y_test,predicted)
