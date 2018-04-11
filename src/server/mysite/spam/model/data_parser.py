import glob
import random
import re
from collections import Counter

import os
import numpy as np

from nltk import ngrams


# Read DATA for SMSSpamCollection
def parse_raw_input(file_name):
    examples_list = []

    with open(file_name, encoding="latin-1") as f:
        raw = f.read()
        examples = raw.split('\n')
        for example in examples:
            spliced_example = example.split('\t')
            if len(spliced_example) >= 2:
                label = example.split('\t')[0]
                content = example.split('\t')[1]
                examples_list.append((label, content))

    return examples_list


# Generate training data and testing data, via
# General sampling
def split_test_train_data(examples_list, percent_test):
    numTest = int(percent_test * len(examples_list))

    # Use set here to remove duplicate examples
    testing_dataset = set(random.sample(examples_list, numTest))
    training_dataset = set(x for x in examples_list if x not in testing_dataset)

    print(len(examples_list))
    print(len(testing_dataset))
    print(len(training_dataset))

    return training_dataset, testing_dataset


# Generate training data and testing data, via
# fold_num-fold sampling
def split_test_train_data_with_folds(examples_list, fold_num):
    folds = []
    fold_block_size = int(len(examples_list) / fold_num)

    random.shuffle(examples_list)

    for i in range(fold_num):
        testing_dataset = set(examples_list[i * fold_block_size:(i + 1) * fold_block_size])
        training_dataset = set(examples_list[0: i * fold_block_size] + examples_list[(i + 1) * fold_block_size:])
        folds.append((training_dataset, testing_dataset))

    return folds


# Create bag of words based on input of list of examples
def create_bag_of_words(training_list, cutoff_frequency):
    rawBagOfWords = []
    regex = re.compile("X-Spam.*\n")

    for label, raw in training_list:
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        for token in tokens:
            rawBagOfWords.append(token)

    # throw out low freq words
    freqDist = Counter(rawBagOfWords)
    bag_of_words = []
    for word, freq in freqDist.items():
        if freq > cutoff_frequency:
            bag_of_words.append(word)

    print(len(bag_of_words))

    return bag_of_words


# For now, set n==2
def create_bag_of_n_grams(training_list, cutoff_frequency, n):
    raw_bag_of_grams = []
    regex = re.compile("X-Spam.*\n")

    for label, raw in training_list:
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        bi_grams = ngrams(tokens, 2)
        for grams in bi_grams:
            raw_bag_of_grams.append(grams)

    freqDist = Counter(raw_bag_of_grams)
    bag_of_grams = []
    for word, freq in freqDist.items():
        if freq > cutoff_frequency:
            bag_of_grams.append(word)

    print(len(bag_of_grams))
    return bag_of_grams

# Generate the feature matrix based on both unigram and ngram
# For now, only use 2-gram
def generate_matrix_ngram(data_list, ngram_dict):
    featureMatrix = np.zeros(shape=(len(data_list),
                                    len(ngram_dict)),
                             dtype=float)
    labelMatrix = np.zeros(shape=(len(data_list), 2), dtype=int)
    regex = re.compile("X-Spam.*\n")

    for i, (label, raw) in enumerate(data_list):
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        fileUniDist = Counter(tokens)
        fileUniNGramDist = Counter(ngrams(tokens, 2))
        for key, value in fileUniNGramDist.items():
            if key in ngram_dict:
                featureMatrix[i, ngram_dict[key]] = value

        if label == 'spam':
            labelMatrix[i, :] = np.array([1, 0])
        else:
            labelMatrix[i, :] = np.array([0, 1])

    return labelMatrix, regularize_matrix(featureMatrix)


# Generate the matrix of Math. form for model training
def generate_matrix(data_list, unigram_dict):
    featureMatrix = np.zeros(shape=(len(data_list),
                                    len(unigram_dict)),
                             dtype=float)
    labelMatrix = np.zeros(shape=(len(data_list), 2), dtype=int)

    regex = re.compile("X-Spam.*\n")
    for i, (label, raw) in enumerate(data_list):
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        fileUniDist = Counter(tokens)
        for key, value in fileUniDist.items():
            if key in unigram_dict:
                featureMatrix[i, unigram_dict[key]] = value
        if label == 'spam':
            labelMatrix[i, :] = np.array([1, 0])
        else:
            labelMatrix[i, :] = np.array([0, 1])

    return labelMatrix, regularize_matrix(featureMatrix)


def regularize_matrix(feature_matrix):
    for doc in range(feature_matrix.shape[0]):
        totalWords = np.sum(feature_matrix[doc, :], axis=0)
        if totalWords > 0:
            feature_matrix[doc, :] = np.multiply(feature_matrix[doc, :], (1 / totalWords))
    return feature_matrix


if __name__ == '__main__':
    datasetFilename = '/Users/jianyuzuo/Workspaces/CSCE665_project/tensorflow-server/src/server/smsdata/SMSSpamCollection'
    examples = parse_raw_input(datasetFilename)
    train, test = split_test_train_data(examples, .1)
    # folds = split_test_train_data_with_folds(examples, 10)

    bagOfWords = create_bag_of_words(train, 5)
    bagOfBiGrams = create_bag_of_n_grams(train, 5, 2)

    # Generate features, this part can be replaced by n-gram and many other features
    # For now, these features are just frequency of words
    features = set(bagOfWords)
    featureDict = {feature: i for i, feature in enumerate(features)}
    # print(len(featureDict))

    biGramFeatures = set(bagOfBiGrams)
    biGramFeatureDict = {feature: i for i, feature in enumerate(biGramFeatures)}
    # print(len(biGramFeatureDict))

    trainY, trainX = generate_matrix(train, featureDict)
    testY, testX = generate_matrix(test, featureDict)

    biTrainY, biTrainX = generate_matrix_ngram(train, biGramFeatureDict)
    biTestY, biTestX = generate_matrix_ngram(test, biGramFeatureDict)

    combinedTrainX = np.concatenate((trainX, biTrainX), axis=1)
    combinedTestX = np.concatenate((testX, biTestX), axis=1)

    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)

    np.savetxt(os.getcwd() + "/data/trainX.csv", trainX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/trainY.csv", trainY, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/testX.csv", testX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/testY.csv", testY, delimiter="\t")

    np.savetxt(os.getcwd() + "/data/biTrainX.csv", combinedTrainX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/biTrainY.csv", biTrainY, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/biTestX.csv", combinedTestX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/biTestY.csv", biTestY, delimiter="\t")

    np.savetxt(os.getcwd() + "/data/gramTrainX.csv", biTrainX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/gramTrainY.csv", biTrainY, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/gramTestX.csv", biTestX, delimiter="\t")
    np.savetxt(os.getcwd() + "/data/gramTestY.csv", biTestY, delimiter="\t")

