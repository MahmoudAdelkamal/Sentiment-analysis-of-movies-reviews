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

def prepare_dataset():

    positive_dataset, negative_dataset = load_dataset()
    positive_dataset = preprocess(positive_dataset)
    positive_dataset = pd.DataFrame(positive_dataset)
    positive_dataset.columns = ['Review']
    positive_dataset['Label'] = ["pos" for i in range(1000)]
    negative_dataset = preprocess(negative_dataset)
    negative_dataset = pd.DataFrame(negative_dataset)
    negative_dataset.columns = ['Review']
    negative_dataset['Label'] = ["neg" for i in range(1000)]
    dataset = pd.concat([positive_dataset,negative_dataset],ignore_index=True,sort=False)
    dataset = shuffle(dataset)
    return dataset

dataset = prepare_dataset()
# dataset statistics
print(dataset.head(5))
print('columns name : ',list(dataset.columns))
print('The size of dataset is : ',len(dataset))
# training
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