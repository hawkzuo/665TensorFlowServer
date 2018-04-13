
# Script used for parsing comments
import os
import pickle
import random
import re
import numpy as np
from collections import Counter

from nltk import ngrams


def parse_raw_input(root_path, file_names):
    examples_list = []

    for file_name in file_names:
        with open(root_path + file_name, encoding="latin-1") as f:
            raw = f.read()
            examples = raw.split('\n')
            for example in examples:
                spliced_example = example.split(',')
                if len(spliced_example) >= 5 and spliced_example[4] != 'CLASS':
                    label = spliced_example[4]
                    content = spliced_example[3]
                    examples_list.append((label, content))

    return examples_list


def split_test_train_data(examples_list, percent_test):
    numTest = int(percent_test * len(examples_list))

    # Use set here to remove duplicate examples
    testing_dataset = set(random.sample(examples_list, numTest))
    training_dataset = set(x for x in examples_list if x not in testing_dataset)

    print('Total DS len: ' + str(len(examples_list)))
    print('Test DS len:' +  str(len(testing_dataset)))
    print('Train DS len:' + str(len(training_dataset)))

    return training_dataset, testing_dataset

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

    # print(len(bag_of_grams))
    return bag_of_grams

def generate_matrix_ngram(data_list, ngram_dict):
    feature_matrix = np.zeros(shape=(len(data_list),
                                     len(ngram_dict)),
                              dtype=float)
    label_matrix = np.zeros(shape=(len(data_list), 2), dtype=int)
    regex = re.compile("X-Spam.*\n")

    for i, (label, raw) in enumerate(data_list):
        raw = re.sub(regex, '', raw)
        # Remove Extra Whitespaces such as '\t' '\n\n\n', etc.
        # TODO: determine whether or not to remove all the ':' '!' '?' symbols
        tokens = " ".join(raw.split()).split()
        fileUniNGramDist = Counter(ngrams(tokens, 2))
        for key, value in fileUniNGramDist.items():
            if key in ngram_dict:
                feature_matrix[i, ngram_dict[key]] = value

        if label == '1':
            label_matrix[i, :] = np.array([1, 0])
        else:
            label_matrix[i, :] = np.array([0, 1])

    return label_matrix, regularize_matrix(feature_matrix)


# Generate the matrix of Math. form for model training
def generate_matrix(data_list, unigram_dict):
    feature_matrix = np.zeros(shape=(len(data_list),
                                     len(unigram_dict)),
                              dtype=float)
    label_matrix = np.zeros(shape=(len(data_list), 2), dtype=int)

    regex = re.compile("X-Spam.*\n")
    for i, (label, raw) in enumerate(data_list):
        raw = re.sub(regex, '', raw)
        # Remove Extra Whitespaces such as '\t' '\n\n\n', etc.
        # TODO: determine whether or not to remove all the ':' '!' '?' symbols
        tokens = " ".join(raw.split()).split()
        fileUniDist = Counter(tokens)
        for key, value in fileUniDist.items():
            if key in unigram_dict:
                feature_matrix[i, unigram_dict[key]] = value
        if label == '1':
            label_matrix[i, :] = np.array([1, 0])
        else:
            label_matrix[i, :] = np.array([0, 1])

    return label_matrix, regularize_matrix(feature_matrix)


def regularize_matrix(feature_matrix):
    for doc in range(feature_matrix.shape[0]):
        totalWords = np.sum(feature_matrix[doc, :], axis=0)
        if totalWords > 0:
            feature_matrix[doc, :] = np.multiply(feature_matrix[doc, :], (1 / totalWords))
    return feature_matrix




if __name__ == '__main__':
    files_root_path = '/Users/jianyuzuo/Workspaces/CSCE665_project/tensorflow-server/src/server/smsdata/YouTube-Spam-Collection-v1/'
    file_names = ['Youtube01-Psy.csv',
                  'Youtube02-KatyPerry.csv',
                  'Youtube03-LMFAO.csv',
                  'Youtube04-Eminem.csv',
                  'Youtube05-Shakira.csv']

    examples = parse_raw_input(files_root_path, file_names)
    train, test = split_test_train_data(examples, .1)

    bagOfWords = create_bag_of_words(train, 5)
    bagOfBiGrams = create_bag_of_n_grams(train, 5, 2)

    features = set(bagOfWords)
    featureDict = {feature: i for i, feature in enumerate(features)}
    biGramFeatures = set(bagOfBiGrams)
    biGramFeatureDict = {feature: i for i, feature in enumerate(biGramFeatures)}

    trainY, trainX = generate_matrix(train, featureDict)
    testY, testX = generate_matrix(test, featureDict)

    biTrainY, biTrainX = generate_matrix_ngram(train, biGramFeatureDict)
    biTestY, biTestX = generate_matrix_ngram(test, biGramFeatureDict)

    combinedTrainX = np.concatenate((trainX, biTrainX), axis=1)
    combinedTestX = np.concatenate((testX, biTestX), axis=1)

    print(os.getcwd())

#     Save to Local file
    np.savetxt("data/Youtube_trainX.csv", trainX, delimiter="\t")
    np.savetxt("data/Youtube_trainY.csv", trainY, delimiter="\t")
    np.savetxt("data/Youtube_testX.csv", testX, delimiter="\t")
    np.savetxt("data/Youtube_testY.csv", testY, delimiter="\t")

    np.savetxt("data/Youtube_biTrainX.csv", combinedTrainX, delimiter="\t")
    np.savetxt("data/Youtube_biTrainY.csv", biTrainY, delimiter="\t")
    np.savetxt("data/Youtube_biTestX.csv", combinedTestX, delimiter="\t")
    np.savetxt("data/Youtube_biTestY.csv", biTestY, delimiter="\t")

    np.savetxt("data/Youtube_gramTrainX.csv", biTrainX, delimiter="\t")
    np.savetxt("data/Youtube_gramTrainY.csv", biTrainY, delimiter="\t")
    np.savetxt("data/Youtube_gramTestX.csv", biTestX, delimiter="\t")
    np.savetxt("data/Youtube_gramTestY.csv", biTestY, delimiter="\t")

    f1 = open('data/Youtube_uniFeature.pickle', 'wb')
    pickle.dump(featureDict, f1)
    f1.close()
    f2 = open('data/Youtube_biFeature.pickle' ,'wb')
    pickle.dump(biGramFeatureDict, f2)
    f2.close()
