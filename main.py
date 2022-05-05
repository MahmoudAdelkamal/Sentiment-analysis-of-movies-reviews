from pre_processing import  *
from Feature_Extraction import *
from models import *


positive_dataset, negative_dataset = load_dataset()
preprocess(positive_dataset)
positive_dataset = pd.DataFrame(positive_dataset)
positive_dataset.columns = ['Review']
positive_dataset['Label'] = ["pos" for i in range(1000)]

preprocess(negative_dataset)
negative_dataset = pd.DataFrame(negative_dataset)
negative_dataset.columns = ['Review']
negative_dataset['Label'] = ["neg" for i in range(1000)]

dataset = pd.concat([positive_dataset,negative_dataset],ignore_index=True,sort=False)

kf=KFold(n_splits=20,random_state=1,shuffle=True)
for train_index,test_index in kf.split(dataset):
    x = dataset['Review']
    y = dataset['Label']
    x_train , x_test, y_train , y_test= x[train_index],x[test_index],y[train_index],y[test_index]
    xTrain = feature_extraction_TF_IDF(x_train)
    yTrain = y_prepaation(y_train)
    yTest = y_prepaation(y_test)
    DT_model = train_Decition_Tree(xTrain,yTrain)
    accuracy = Accuracy_Score(x_test,yTest,model)
