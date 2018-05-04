import csv
import os
import pickle
import random
from collections import Counter

import numpy as np
from nltk import ngrams

from server.mysite.spam.model.youtube.util.content_processor import generate_tokens_from_parsed_soup_text

# Handling some other charsets such as:
# This script mainly removes the headers and gathers
# useful information from data source and save to text file
# Insights: only do UTF-8 encoding data
# Reason: Consistency ++ other charset help little between each other

DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/youtube/'
DATA_PREFIX_WIN = 'D:\\workplaces\\665\\youtube\\'

CRAWLED_DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/crawler/'
CRAWLED_DATA_PREFIX_WIN = 'D:\\workplaces\\665\\crawler\\'


def load_dataset_data_youtube(is_win_platform=False):
    if is_win_platform:
        files = os.listdir(DATA_PREFIX_WIN)
        root_prefix = DATA_PREFIX_WIN
    else:
        root_prefix = DATA_PREFIX
        files = os.listdir(DATA_PREFIX)

    examples_list = []
    for file in files:
        with open(root_prefix + file, encoding="latin-1") as f:
            spam_reader = csv.reader(f, delimiter=',', quotechar='"')
            step = []
            for row in spam_reader:
                label = row[4]
                content = row[3]
                tokens, url_count = generate_tokens_from_parsed_soup_text(content)
                step.append((label, " ".join(tokens), url_count))
            step.pop(0)
            examples_list += step
    return examples_list


def load_crawler_data_youtube():
    pass


def generate_model_in_memory(is_win_platform=False,
                             uni_cutoff=8,
                             bi_cutoff=14,
                             split=.2,
                             file_prefix='',
                             operational_mode=1):
    total_examples = load_dataset_data_youtube(is_win_platform)
    trainList, testList = split_test_train_data(total_examples, split)

    uniFeatures = create_bag_of_words(trainList, uni_cutoff)
    uniFeatureDict = {feature: i for i, feature in enumerate(uniFeatures)}

    bagOfBiGrams = create_bag_of_n_grams(trainList, bi_cutoff, 2)
    biGramFeatures = set(bagOfBiGrams)
    biGramFeatureDict = {feature: i for i, feature in enumerate(biGramFeatures)}

    print('n-gram dictionary created')
    uniTrainY, uniTrainX = generate_matrix(trainList, uniFeatureDict)
    uniTestY, uniTestX = generate_matrix(testList, uniFeatureDict)

    biTrainY, biTrainX = generate_matrix_ngram(trainList, biGramFeatureDict, 2)
    biTestY, biTestX = generate_matrix_ngram(testList, biGramFeatureDict, 2)

    if operational_mode == 1:
        combinedTrainX = uniTrainX
        combinedTestX = uniTestX
    elif operational_mode == 2:
        combinedTrainX = np.concatenate((uniTrainX, biTrainX), axis=1)
        combinedTestX = np.concatenate((uniTestX, biTestX), axis=1)
    else:
        raise ValueError('operational_mode', 'not supported')
    print('Storing features dictionary to location:', os.getcwd())

    if is_win_platform:
        feature_prefix = 'features\\'+file_prefix
    else:
        feature_prefix = 'features/'+file_prefix
    with open(feature_prefix + '-uniFeatureDict.pickle', 'wb') as f:
        pickle.dump(uniFeatureDict, f)
    with open(feature_prefix + '-biGramFeatureDict.pickle', 'wb') as f:
        pickle.dump(biGramFeatureDict, f)
    print('Finished saving features file')
    return combinedTrainX, biTrainY, combinedTestX, biTestY


def split_test_train_data(examples_list, ratio):
    numTest = int(ratio * len(examples_list))
    testing_dataset = set(random.sample(examples_list, numTest))
    training_dataset = set(x for x in examples_list if x not in testing_dataset)

    print('Total examples', len(examples_list))
    print('Test examples', len(testing_dataset))
    print('Train examples', len(training_dataset))

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

    for _, combined, _ in training_list:
        for token in combined.split():
            rawBagOfWords.append(token)

    # throw out low freq words
    freqDist = Counter(rawBagOfWords)
    bag_of_words = []
    for word, freq in freqDist.items():
        if freq > cutoff_frequency:
            bag_of_words.append(word)

    print('Length of Bag of words: ', len(bag_of_words))

    return bag_of_words


def create_bag_of_n_grams(training_list, cutoff_frequency, n):
    raw_bag_of_grams = []

    for _, combined, _ in training_list:
        tokens = combined.split()
        bi_grams = ngrams(tokens, n)
        for grams in bi_grams:
            raw_bag_of_grams.append(grams)

    freqDist = Counter(raw_bag_of_grams)
    bag_of_grams = []
    for word, freq in freqDist.items():
        if freq > cutoff_frequency:
            bag_of_grams.append(word)

    print('Length of ', n, 'gram words: ', len(bag_of_grams))
    return bag_of_grams


# Generate the matrix of Math. form for model training
def generate_matrix(data_list, unigram_dict):
    feature_matrix = np.zeros(shape=(len(data_list),
                                     len(unigram_dict)),
                              dtype=float)
    label_matrix = np.zeros(shape=(len(data_list), 2), dtype=int)

    for i, (label, combined, url_count) in enumerate(data_list):
        tokens = combined.split()
        fileUniDist = Counter(tokens)
        for key, value in fileUniDist.items():
            if key in unigram_dict:
                feature_matrix[i, unigram_dict[key]] = value
        if label == '1':  # spam
            label_matrix[i, :] = np.array([1, 0])
        else:
            label_matrix[i, :] = np.array([0, 1])

    return label_matrix, regularize_matrix(feature_matrix)


# Generate the feature matrix based on both unigram and ngram
def generate_matrix_ngram(data_list, ngram_dict, n):
    feature_matrix = np.zeros(shape=(len(data_list),
                                     len(ngram_dict)),
                              dtype=float)
    label_matrix = np.zeros(shape=(len(data_list), 2), dtype=int)

    for i, (label, combined, url_count) in enumerate(data_list):
        tokens = combined.split()
        fileUniNGramDist = Counter(ngrams(tokens, n))
        for key, value in fileUniNGramDist.items():
            if key in ngram_dict:
                feature_matrix[i, ngram_dict[key]] = value

        if label == '1':  # spam
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
    # examples = load_dataset_data_youtube(True)
    a, b, c, d = generate_model_in_memory(True)

    print('End')
    pass
