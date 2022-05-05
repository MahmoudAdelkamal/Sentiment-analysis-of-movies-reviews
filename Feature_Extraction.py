from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer


def feaure_extraction_Bag_of_N_grams(x_train):
    cv = CountVectorizer(ngram_range=(2,2))
    features = cv.fit_transform(x_train)
    return features
def feature_extraction_TF_IDF(x_train):
    TF_IDF = TfidfVectorizer(max_df = 0.9,min_df = 2,max_features = 800, stop_words = 'english')
    features = TF_IDF.fit_transform(x_train)
    return features

def y_prepaation(y_):
    Y = []
    for i in y_:
        if str(i) == 'pos':
            Y.append(1)
        else:
            Y.append(0)
    return Y

