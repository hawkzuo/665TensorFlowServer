import os
import pickle
from collections import Counter

import numpy as np
from nltk import ngrams

from .email_processor import generate_tokens_from_parsed_soup_text


def csv_to_numpy_array(filePath, delimiter):
    result = np.genfromtxt(filePath, delimiter=delimiter, dtype=None)
    print(result.shape)
    return result


# Import data of different data types:
# 1: basic Bag of Word features matrix
# 2: Combined features matrix [1-gram 2-gram]
# 3: Combined features matrix [1-gram 2-gram 3-gram]

# We don't save examples to local file
# Due to Memory Error
def import_data(data_type):
    if "data" not in os.listdir(os.getcwd()):
        raise Exception('data', 'not presented')
    else:
        pass

    if data_type == 1:
        print("loading train unigram ")
        train_X = csv_to_numpy_array("data/uniTrainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/uniTrainY.csv", delimiter="\t")
        print("loading test unigram ")
        test_X = csv_to_numpy_array("data/uniTestX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/uniTestY.csv", delimiter="\t")
    elif data_type == 2 or data_type == 3:
        print("loading train combined ")
        train_X = csv_to_numpy_array("data/combinedTrainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/combinedTrainY.csv", delimiter="\t")
        print("loading test combined ")
        test_X = csv_to_numpy_array("data/combinedTestX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/combinedTestY.csv", delimiter="\t")
    else:
        raise Exception('data', 'unsupported type')

    return train_X, train_Y, test_X, test_Y


# Load FeatureDict from disk
# 1: only unigram features
# 2: Combined features matrix [1-gram 2-gram]
# 3: Combined features matrix [1-gram 2-gram 3-gram]
def import_features_dict(data_type):
    if "features" not in os.listdir(os.getcwd()):
        raise Exception('features', 'not presented')
    else:
        pass

    if data_type == 1:
        with open("features/uniFeatureDict.pickle", 'rb') as f:
            uniFeatureDict = pickle.load(f)
        return uniFeatureDict, None, None
    elif data_type == 2:
        with open("features/uniFeatureDict.pickle", 'rb') as f:
            uniFeatureDict = pickle.load(f)
        with open("features/biGramFeatureDict.pickle", 'rb') as f:
            biGramFeatureDict = pickle.load(f)
        return uniFeatureDict, biGramFeatureDict, None
    elif data_type == 3:
        with open("features/uniFeatureDict.pickle", 'rb') as f:
            uniFeatureDict = pickle.load(f)
        with open("features/biGramFeatureDict.pickle", 'rb') as f:
            biGramFeatureDict = pickle.load(f)
        with open("features/triGramFeatureDict.pickle", 'rb') as f:
            triGramFeatureDict = pickle.load(f)
        return uniFeatureDict, biGramFeatureDict, triGramFeatureDict
    else:
        raise Exception('feature', 'unsupported type')


# Load featuresCount labelsCount examplesCount from disk
def import_structure():
    if "features" not in os.listdir(os.getcwd()):
        raise Exception('features', 'not presented')
    else:
        pass
    with open("features/structure.pickle", 'rb') as f:
        struct_dict = pickle.load(f)
    return struct_dict['features'], struct_dict['labels'], struct_dict['examples']

# Generate tokens and url_links count from raw input
def tokenize_raw_input(raw_input):
    web_tokens, _ = generate_tokens_from_parsed_soup_text(raw_input)
    return web_tokens

# Generate Unigram features for testing example
def generate_sample_unigram(tokens, unigram_dict):
    feature_matrix = np.zeros(shape=(1, len(unigram_dict)), dtype=float)

    fileUniDist = Counter(tokens)
    for key, value in fileUniDist.items():
        if key in unigram_dict:
            feature_matrix[0, unigram_dict[key]] = value
    return regularize_matrix(feature_matrix)

# Generate Ngram features for testing example
def generate_sample_ngram(tokens, ngram_dict, n):
    feature_matrix = np.zeros(shape=(1, len(ngram_dict)), dtype=float)

    fileUniNGramDist = Counter(ngrams(tokens, n))
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

    pass

