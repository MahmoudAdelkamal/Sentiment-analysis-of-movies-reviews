from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier as DT
from sklearn.model_selection import train_test_split
import random
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
kf=KFold(n_splits=10,random_state=1,shuffle=True)
Logistic_regression_accuracy = 0
for train_index,test_index in kf.split(dataset):
    x = dataset['Review']
    y = dataset['Label']
    x_train , x_test, y_train , y_test= x[train_index],x[test_index],y[train_index],y[test_index]
    print(len(x_test))
    clf = train_Logistic_Regression(x_train,y_train)
    Logistic_regression_accuracy+= Accuracy_Score(x_test,y_test,clf)

print((Logistic_regression_accuracy/10)*100,'%')