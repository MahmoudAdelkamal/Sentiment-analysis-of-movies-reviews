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

def Accuracy_Score(x_test,y_test,model):
    predicted= model.predict(x_test)
    return accuracy_score(predicted,y_test)

def ConfusionMatrix(x_test,y_test,model) :
    predicted= model.predict(x_test)
    cf_matrix = confusion_matrix(y_test,predicted)
    sns.set()
    sns.heatmap(cf_matrix,annot=True,annot_kws={"size" : 16})
    plt.show()