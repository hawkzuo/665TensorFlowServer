import os
import pickle

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

# This format is acceptable for the Django Settings
from .util import FeatureExtraction
# This format is acceptable for single running purposes
# from server.mysite.spam.model.gmail import data_helper_gmail

DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'
SPAM_PREFIX = 'spamout/m'
MONTH_MAPPING = ['', '01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']

operationalMode = 3

uniFeatureDict, biGramFeatureDict, triGramFeatureDict = FeatureExtraction.import_features_dict(operationalMode)
numFeatures, numLabels, numTrainExamples = import_structure()

timeSteps = 1
hiddenUnits = 512
desiredBatches = 60

learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                          global_step=1,
                                          decay_steps=numTrainExamples,
                                          decay_rate=0.95,
                                          staircase=True)
batchSize = numTrainExamples // desiredBatches

# weights biases
outWeights = tf.Variable(tf.random_normal([hiddenUnits, numLabels]))
outBias = tf.Variable(tf.random_normal([numLabels]))

# X ~ N*M,
# N: # of examples, M:# of features
x = tf.placeholder(tf.float32, [None, timeSteps, numFeatures])
# tY ~ N*C, tY short for "true Y"
# C: # of classes
y = tf.placeholder(tf.float32, [None, numLabels])

input = tf.unstack(x, timeSteps, 1)

lstm_layer = rnn.BasicLSTMCell(hiddenUnits, forget_bias=1)

outputs, _ = rnn.static_rnn(lstm_layer, input, dtype="float32")

# converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
prediction = tf.matmul(outputs[-1], outWeights) + outBias

# loss_function
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
# optimization
opt = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss)

# model evaluation
correct_prediction = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

# load variables from file
saver = tf.train.Saver()
print("Load weights from directory: " + str(os.getcwd()) + "/weights_lstm")
saver.restore(sess, os.getcwd() + "/weights_lstm/lstm_mode" + str(operationalMode) + "_trained_variables.ckpt")


def next_batch(batch_size, batch_id, X, Y):
    begin_index = (batch_id * batch_size) % (X.shape[0])
    if begin_index + batch_size < X.shape[0]:
        end_index = begin_index + batch_size
    else:
        end_index = X.shape[0]
    # test[numpy.logical_or.reduce([test[:, 1] == x for x in wanted])]
    b_x = X[begin_index:end_index, :]
    b_y = Y[begin_index:end_index, :]

    return b_x, b_y


# Two functions used for server
def predict_from_raw_input(raw_input, operation_mode=3):
    tokens, _ = generate_tokens_from_parsed_soup_text(raw_input)
    uniMatrix = generate_sample_unigram(tokens, uniFeatureDict)
    biMatrix = generate_sample_ngram(tokens, biGramFeatureDict, 2)
    triMatrix = generate_sample_ngram(tokens, triGramFeatureDict, 3)

    if operation_mode == 1:
        combinedMatrixX = uniMatrix
    elif operation_mode == 2:
        combinedMatrixX = np.concatenate((uniMatrix, biMatrix), axis=1)
    elif operation_mode == 3:
        combinedMatrixX = np.concatenate((uniMatrix, biMatrix, triMatrix), axis=1)
    else:
        raise ('operation_mode', 'unsupported')

    tensor_prediction = sess.run(prediction, feed_dict={x: combinedMatrixX.reshape(1, timeSteps,
                                                                                   numFeatures)})  # had to make sure that each input in feed_dict was an array
    print(tensor_prediction)
    single_prediction = single_label_to_string(tensor_prediction)
    print(single_prediction)
    return single_prediction


# This translate tensor prediction into readable label
def single_label_to_string(tensor_prediction):
    # 2nd vector value > 1st vector value => [0,1] => 'ham'
    if tensor_prediction[0][1] > tensor_prediction[0][0]:
        return 'ham'
    else:
        return 'spam'


### Those are used for evaluation on this predictor


def read_spams_and_generate_features_for_month_year(month, year, operation_mode):
    spam_location = DATA_PREFIX + SPAM_PREFIX + str(year) + MONTH_MAPPING[month]
    filenames = os.listdir(spam_location)

    # for f in filenames:
    #     if 'pickle' in f:
    #         pass
    #     else:
    #         print(f)

    examples_count = len(filenames)
    errors_count = 0

    print('Spams on year', year, 'month', month, 'is', examples_count)

    for i in range(examples_count):
        if i % 100 == 0:
            print(errors_count, '/', i)
        # File name start with 1
        with open(spam_location + '/m' + str(i + 1) + '.pickle', 'rb') as f:
            single_example = pickle.load(f)
            tokens = single_example['content']

            uniMatrix = generate_sample_unigram(tokens, uniFeatureDict)
            biMatrix = generate_sample_ngram(tokens, biGramFeatureDict, 2)
            triMatrix = generate_sample_ngram(tokens, triGramFeatureDict, 3)
            if operation_mode == 1:
                combinedMatrixX = uniMatrix
            elif operation_mode == 2:
                combinedMatrixX = np.concatenate((uniMatrix, biMatrix), axis=1)
            elif operation_mode == 3:
                combinedMatrixX = np.concatenate((uniMatrix, biMatrix, triMatrix), axis=1)
            else:
                raise ('operation_mode', 'unsupported')

            tensor_prediction = sess.run(prediction, feed_dict={x: combinedMatrixX.reshape(1, timeSteps,
                                                                                           numFeatures)})  # had to make sure that each input in feed_dict was an array
            single_prediction = single_label_to_string(tensor_prediction)
            # print(single_prediction)
            if single_prediction == 'ham':
                errors_count += 1
    print(errors_count)
    print('Performance on year', year, 'month', month, 'is', errors_count / examples_count)
    return examples_count


def read_spams_for_year(year, operation_mode):
    year_list = []
    for i in range(12):
        year_list += read_spams_and_generate_features_for_month_year(i + 1, year, operation_mode)
    print('Spams on whole year', year, 'is', len(year_list))
    return year_list


if __name__ == "__main__":
    # raw1 = "I thought slide is enough."

    raw1 = """
    abc
    """

    print(predict_from_raw_input(raw1))

    # read_spams_and_generate_features_for_month_year(10, 2017, operationalMode)
    # read_spams_and_generate_features_for_month_year(11, 2017, operationalMode)
    # read_spams_and_generate_features_for_month_year(12, 2017, operationalMode)

    pass
