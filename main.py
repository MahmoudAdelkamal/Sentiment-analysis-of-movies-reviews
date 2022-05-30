from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
import random
import matplotlib.pyplot as plt
import seaborn as sns
from Evaluation import *
from sklearn.utils import shuffle
from sklearn.metrics import classification_report
from pre_processing import  *
from Models import *

def dataset_statistics(dataset) :

    print(dataset.head(5))
    print(dataset.info())
    sns.set()
    sns.countplot(dataset['Label'], color='blue')
    plt.show()
def kfold_cross_validation(dataset) :

    kf=KFold(n_splits=10,random_state=1,shuffle=True)
    Logistic_regression_accuracy = []
    svm_accuracy = []
    for train_index,test_index in kf.split(dataset):
        x = dataset['Review']
        y = dataset['Label']
        x_train , x_test, y_train , y_test= x[train_index],x[test_index],y[train_index],y[test_index]
        clf_logistic = train_Logistic_Regression(x_train,y_train)
        clf_svm = train_svc(x_train,y_train)
        Logistic_regression_accuracy.append(Accuracy_Score(x_test,y_test,clf_logistic))
        svm_accuracy.append(Accuracy_Score(x_test,y_test,clf_svm))
    return Logistic_regression_accuracy,svm_accuracy,x_test,y_test,clf_logistic,clf_svm


dataset = prepare_dataset()
dataset_statistics(dataset)
shuffle(dataset)
Logistic_regression_accuracy, svm_accuracy, x_test, y_test, clf_logistic, clf_svm = kfold_cross_validation(dataset)

#logistic regression model
print((sum(Logistic_regression_accuracy)/len(Logistic_regression_accuracy))*100,'%')
ConfusionMatrix(x_test,y_test,clf_logistic)

# svc model
print((sum(svm_accuracy)/len(svm_accuracy))*100,'%')
print(ConfusionMatrix(x_test,y_test,clf_svm))