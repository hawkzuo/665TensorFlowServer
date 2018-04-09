import glob
import random
import re
from collections import Counter
from itertools import filterfalse

import numpy as np


# Read DATA for SMSSpamCollection
def parse_raw_input(file_name):
    spam_list = []
    ham_list = []

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
def create_bag_of_words(ham_list, spam_list):
    bagOfWords = []
    regex = re.compile("X-Spam.*\n")

    for label, raw in ham_list:
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        for token in tokens:
            bagOfWords.append(token)
    for label, raw in spam_list:
        raw = re.sub(regex, '', raw)
        tokens = raw.split()
        for token in tokens:
            bagOfWords.append(token)

    print(len(bagOfWords))
    return bagOfWords


if __name__ == '__main__':
    dataset_filename = '/Users/jianyuzuo/Workspaces/CSCE665_project/tensorflow-server/src/server/smsdata/SMSSpamCollection'
    examples = parse_raw_input(dataset_filename)
    # train, test = split_test_train_data(examples, .1)

    folds = split_test_train_data_with_folds(examples, 10)

    # trainX,trainY,testX,testY = reader.input_data(hamDir=hamDir,
    #                                               spamDir=spamDir,
    #                                               percentTest=.1,
    #                                               cutoff=15)
