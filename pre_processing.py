import os
import re
import pandas as pd
import nltk
from sklearn.model_selection import KFold
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.porter import PorterStemmer

def read_text_file(file_path):

    file = ""
    with open(file_path, 'r') as f:
        file+=f.read()
    return file

def read_files(path):

    files = []
    for file in os.listdir():
        if file.endswith(".txt"):
            file_path = f"{path}\{file}"
            files.append(read_text_file(file_path))
    return files

def load_dataset() :

    negative_dataset_path = "D:\college material\\4th year\\2nd term\\NLP\\Sentiment-analysis-of-movies-reviews\\txt_sentoken\\neg"
    positive_dataset_path = "D:\college material\\4th year\\2nd term\\NLP\\Sentiment-analysis-of-movies-reviews\\txt_sentoken\\pos"
    os.chdir(negative_dataset_path)
    negative_dataset = read_files(negative_dataset_path)
    os.chdir(positive_dataset_path)
    positive_dataset = read_files(positive_dataset_path)

    return positive_dataset,negative_dataset

def remove_stop_words(dataset):

    stop_words = set(stopwords.words("english"))
    for i in range(len(dataset)):
        dataset[i] = [word for word in dataset[i] if word.casefold() not in stop_words]
    return dataset

def remove_numbers(dataset):
    for i in range(len(dataset)):
        dataset[i] = re.sub("\d+",' ', dataset[i])
        #dataset[i] = word_tokenize(dataset[i])
    return dataset

def remove_punctuation(dataset):

    for i in range(len(dataset)):
        dataset[i] = re.sub("[^a-zA-z | ^\w+'t]",' ', dataset[i])
        dataset[i] = word_tokenize(dataset[i])
    return dataset

def stemming(dataset):

    porter_stemmer = PorterStemmer()
    for i in range(len(dataset)):
        dataset[i] = [porter_stemmer.stem(word) for word in dataset[i]]
        dataset[i] = ' '.join(dataset[i])
    return dataset

def preprocess(dataset):

    dataset = remove_numbers(dataset)
    dataset = remove_punctuation(dataset)
    dataset = remove_stop_words(dataset)
    dataset = stemming(dataset)
    return dataset
