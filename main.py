from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
import random
from Evaluation import *
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from pre_processing import  *
from Models import *

def dataset_statistics(dataset) :

    print(dataset.head(5))
    print('columns name : ', list(dataset.columns))
    print('The size of dataset is : ', len(dataset))

dataset = prepare_dataset()
dataset_statistics(dataset)
kf=KFold(n_splits=10,random_state=1,shuffle=True)
Logistic_regression_accuracy = []
svm_accuracy = []
for train_index,test_index in kf.split(dataset):
    x = dataset['Review']
    y = dataset['Label']
    x_train , x_test, y_train , y_test= x[train_index],x[test_index],y[train_index],y[test_index]
    clf = train_Logistic_Regression(x_train,y_train)
    clf_svm = train_svc(x_train,y_train)
    Logistic_regression_accuracy.append(Accuracy_Score(x_test,y_test,clf))
    svm_accuracy.append(Accuracy_Score(x_test,y_test,clf_svm))
#logistic regression model
print((sum(Logistic_regression_accuracy)/len(Logistic_regression_accuracy))*100,'%')
print(classificationReport(x_test,y_test,clf))
print(ConfusionMatrix(x_test,y_test,clf))
# svc model
print((sum(svm_accuracy)/len(svm_accuracy))*100,'%')
print(classificationReport(x_test,y_test,clf_svm))
print(ConfusionMatrix(x_test,y_test,clf_svm))