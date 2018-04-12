import os
import pickle
import re
from collections import Counter

import numpy as np
from nltk import ngrams


def csv_to_numpy_array(filePath, delimiter):
    result = np.genfromtxt(filePath, delimiter=delimiter, dtype=None)
    print(result.shape)
    return result


# Import data of different data types:
# 1: basic Bag of Word features matrix
# 2: Plus Character-level-n-gram features matrix
# 3: only n-gram features

def import_data(data_type):
    if "data" not in os.listdir(os.getcwd()):
        raise Exception('data', 'not presented')
    else:
        pass

    if data_type == 1:
        print("loading training data for unigram mode")
        train_X = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
        print("loading test data for unigram mode")
        test_X = csv_to_numpy_array("data/testX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    elif data_type == 2:
        print("loading training data for combined[1,2-gram] mode")
        train_X = csv_to_numpy_array("data/biTrainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/biTrainY.csv", delimiter="\t")
        print("loading test data for combined[1,2-gram] mode")
        test_X = csv_to_numpy_array("data/biTestX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/biTestY.csv", delimiter="\t")
    elif data_type == 3:
        print("loading training data for bi-gram mode")
        train_X = csv_to_numpy_array("data/gramTrainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/gramTrainY.csv", delimiter="\t")
        print("loading test data for bi-gram mode")
        test_X = csv_to_numpy_array("data/gramTestX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/gramTestY.csv", delimiter="\t")
    else:
        raise Exception('data', 'unsupported type')

    return train_X, train_Y, test_X, test_Y

# Load FeatureDict from disk
# 1: only unigram features
# 2: unigram & n-gram features
# 3: only n-gram features
def import_features_dict(data_type):
    if "data" not in os.listdir(os.getcwd()):
        raise Exception('data', 'not presented')
    else:
        pass

    if data_type == 1:
        f1 = open("data/uniFeature.pickle", 'rb')
        uniFeatureDict = pickle.load(f1)
        f1.close()
        return uniFeatureDict, None
    elif data_type == 2:
        f1 = open("data/uniFeature.pickle", 'rb')
        uniFeatureDict = pickle.load(f1)
        f1.close()
        f2 = open("data/biFeature.pickle", 'rb')
        biGramFeatureDict = pickle.load(f2)
        f2.close()

        return uniFeatureDict, biGramFeatureDict
    elif data_type == 3:
        f2 = open("data/biFeature.pickle", 'rb')
        biGramFeatureDict = pickle.load(f2)
        f2.close()
        return None, biGramFeatureDict
    else:
        raise Exception('data', 'unsupported type')



# Generate Unigram features for testing example
def generate_sample_unigram(raw_input, unigram_dict):
    feature_matrix = np.zeros(shape=(1, len(unigram_dict)), dtype=float)
    regex = re.compile("X-Spam.*\n")
    raw = re.sub(regex, '', raw_input)
    tokens = " ".join(raw.split()).split()
    fileUniDist = Counter(tokens)
    for key, value in fileUniDist.items():
        if key in unigram_dict:
            feature_matrix[0, unigram_dict[key]] = value
    return regularize_matrix(feature_matrix)


# Generate Ngram features for testing example
def generate_sample_ngram(raw_input, ngram_dict):
    feature_matrix = np.zeros(shape=(1, len(ngram_dict)), dtype=float)
    regex = re.compile("X-Spam.*\n")
    raw = re.sub(regex, '', raw_input)
    tokens = " ".join(raw.split()).split()
    fileUniNGramDist = Counter(ngrams(tokens, 2))
    for key, value in fileUniNGramDist.items():
        if key in ngram_dict:
            feature_matrix[0, ngram_dict[key]] = value
    return regularize_matrix(feature_matrix)

def regularize_matrix(feature_matrix):
    for doc in range(feature_matrix.shape[0]):
        totalWords = np.sum(feature_matrix[doc, :], axis=0)
        if totalWords > 0:
            feature_matrix[doc, :] = np.multiply(feature_matrix[doc, :], (1 / totalWords))
    return feature_matrix



if __name__ == "__main__":
    uD, bD = import_features_dict(2)
    print(len(uD))
    print(len(bD))

