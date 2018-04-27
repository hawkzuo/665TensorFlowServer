# total ham emails :
# ONLY INBOX    ->  8104-2
# Total         ->  12904-2
# Selected spam emails:
# 4 configurations with Ratio: [1:1, 1:2, 1:3, 1:4, 1:5]
# 1 configurations with Source: [Most recent] first
# Total configuration combinations = 10

# Train-Test Split: Either 80-20 or 70-30

import numpy as np
import os
import pickle
import random
from collections import Counter

from nltk import ngrams

ONLY_INBOX_HAM = 1
ALL_HAM = 2
SPAM_FOLDERS = ['/m201804', '/m201803', '/m201802', '/m201801', '/m201712', '/m201711']
SPAM_FOLDERS_COUNT = [3574, 20040, 25948, 38431, 51299, 74722]


def generate_examples_list(ham_path, spam_path, rate):
    ham_filenames = os.listdir(ham_path)
    ham_examples_count = len(ham_filenames)
    ham_examples_list = []

    for i in range(ham_examples_count):
        with open(ham_path + '/m' + str(i) + '.pickle', 'rb') as f:
            single_example = pickle.load(f)
            ham_examples_list.append(('ham', " ".join(single_example['content']), single_example['url']))

    spam_examples_list = []
    spam_examples_count = ham_examples_count * rate
    upper_index = 0
    while upper_index < 6:
        if SPAM_FOLDERS_COUNT[upper_index] > spam_examples_count:
            break
        else:
            upper_index += 1

    # Those folders' examples will all be used
    for j in range(upper_index):
        step_spam_filenames = os.listdir(spam_path + SPAM_FOLDERS[j])
        step_spam_examples_count = len(step_spam_filenames)
        for i in range(1, 1 + step_spam_examples_count):
            with open(spam_path + SPAM_FOLDERS[j] + '/m' + str(i) + '.pickle', 'rb') as f:
                single_example = pickle.load(f)
                spam_examples_list.append(('spam', " ".join(single_example['content']), single_example['url']))
        pass

    all_include_index = upper_index - 1
    spam_leftovers_count = spam_examples_count - SPAM_FOLDERS_COUNT[all_include_index]
    for i in range(1, 1 + spam_leftovers_count):
        with open(spam_path + SPAM_FOLDERS[upper_index] + '/m' + str(i) + '.pickle', 'rb') as f:
            single_example = pickle.load(f)
            spam_examples_list.append(('spam', " ".join(single_example['content']), single_example['url']))

    return spam_examples_list, ham_examples_list


# Data Format: (label, content, urls)
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
        if label == 'spam':
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

        if label == 'spam':
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


# operational_mode:
# 1: unigram;   2: unigram + bigram;    3: unigram + bigram + trigram
def generate_model_in_memory(data_prefix,
                             uni_cutoff=10,
                             bi_cutoff=20,
                             tri_cutoff=20,
                             split=.2,
                             spam_ham_ratio=2,
                             operational_mode=3,
                             file_prefix='',
                             is_cross_validation_mode=False):

    ham_prefix = data_prefix + 'data/parsed_ionly'
    spam_prefix = data_prefix + 'data/most_recent_spam'

    spamList, hamList = generate_examples_list(ham_prefix, spam_prefix, spam_ham_ratio)

    totalList = spamList + hamList

    if is_cross_validation_mode:
        pass
    else:
        trainList, testList = split_test_train_data(totalList, split)

    # TODO: 10-fold cross validation
    # folds = split_test_train_data_with_folds(totalList, 10)

    # Parameter Tuning for cutoff frequency
    # 5 => 19000
    # 10 => 12000
    # 15 => 9199
    # 20 => 7723

    # 5* => 19000+
    uniFeatures = create_bag_of_words(trainList, uni_cutoff)
    uniFeatureDict = {feature: i for i, feature in enumerate(uniFeatures)}

    bagOfBiGrams = create_bag_of_n_grams(trainList, bi_cutoff, 2)
    biGramFeatures = set(bagOfBiGrams)
    biGramFeatureDict = {feature: i for i, feature in enumerate(biGramFeatures)}

    bagOfTriGrams = create_bag_of_n_grams(trainList, tri_cutoff, 3)
    triGramFeatures = set(bagOfTriGrams)
    triGramFeatureDict = {feature: i for i, feature in enumerate(triGramFeatures)}

    print('n-gram dictionary created')

    uniTrainY, uniTrainX = generate_matrix(trainList, uniFeatureDict)
    uniTestY, uniTestX = generate_matrix(testList, uniFeatureDict)

    biTrainY, biTrainX = generate_matrix_ngram(trainList, biGramFeatureDict, 2)
    biTestY, biTestX = generate_matrix_ngram(testList, biGramFeatureDict, 2)

    triTrainY, triTrainX = generate_matrix_ngram(trainList, triGramFeatureDict, 3)
    triTestY, triTestX = generate_matrix_ngram(testList, triGramFeatureDict, 3)

    if operational_mode == 1:
        combinedTrainX = uniTrainX
        combinedTestX = uniTestX
    elif operational_mode == 2:
        combinedTrainX = np.concatenate((uniTrainX, biTrainX), axis=1)
        combinedTestX = np.concatenate((uniTestX, biTestX), axis=1)
    elif operational_mode == 3:
        combinedTrainX = np.concatenate((uniTrainX, biTrainX, triTrainX), axis=1)
        combinedTestX = np.concatenate((uniTestX, biTestX, triTestX), axis=1)
    else:
        raise ValueError('operational_mode', 'not supported')
    print('Storing features dictionary to location:', os.getcwd())

    with open('features/'+file_prefix+'uniFeatureDict.pickle', 'wb') as f:
        pickle.dump(uniFeatureDict, f)
    with open('features/'+file_prefix+'biGramFeatureDict.pickle', 'wb') as f:
        pickle.dump(biGramFeatureDict, f)
    with open('features/'+file_prefix+'triGramFeatureDict.pickle', 'wb') as f:
        pickle.dump(triGramFeatureDict, f)

    print('Finished saving features file')

    return combinedTrainX, biTrainY, combinedTestX, biTestY


if __name__ == '__main__':
    # print('main')
    pass