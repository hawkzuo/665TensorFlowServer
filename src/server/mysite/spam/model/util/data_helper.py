import os

import numpy as np


def csv_to_numpy_array(filePath, delimiter):
    result = np.genfromtxt(filePath, delimiter=delimiter, dtype=None)
    print(result.shape)
    return result


# Import data of different Type:
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
