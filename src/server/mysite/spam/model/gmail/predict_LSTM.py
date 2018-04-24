import os

import tensorflow as tf
import numpy as np
from tensorflow.contrib import rnn

from server.mysite.spam.model.gmail import data_helper_gmail

operationalMode = 3

uniFeatureDict, biGramFeatureDict, triGramFeatureDict = data_helper_gmail.import_features_dict(operationalMode)
numFeatures, numLabels, numTrainExamples = data_helper_gmail.import_structure()

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


def predict_from_raw_html(html, operational_mode=3):
    pass


# Two functions used for server
def predict_from_raw_input(raw_input, operation_mode=3):

    tokens, _ = data_helper_gmail.generate_tokens_from_parsed_soup_text(raw_input)
    uniMatrix = data_helper_gmail.generate_sample_unigram(tokens, uniFeatureDict)
    biMatrix = data_helper_gmail.generate_sample_ngram(tokens, biGramFeatureDict, 2)
    triMatrix = data_helper_gmail.generate_sample_ngram(tokens, triGramFeatureDict, 3)

    if operation_mode == 1:
        combinedMatrixX = uniMatrix
    elif operation_mode == 2:
        combinedMatrixX = np.concatenate((uniMatrix, biMatrix), axis=1)
    elif operation_mode == 3:
        combinedMatrixX = np.concatenate((uniMatrix, biMatrix, triMatrix), axis=1)
    else:
        raise('operation_mode', 'unsupported')

    tensor_prediction = sess.run(prediction, feed_dict={x: combinedMatrixX.reshape(1, timeSteps,
                                                                                   numFeatures)})  # had to make sure that each input in feed_dict was an array
    print(tensor_prediction)
    single_prediction = single_label_to_string(tensor_prediction)
    return single_prediction


def single_label_to_string(tensor_prediction):
    # 2nd vector value > 1st vector value => [0,1] => 'ham'
    if tensor_prediction[0][1] > tensor_prediction[0][0]:
        return 'ham'
    else:
        return 'spam'


if __name__ == "__main__":
    # raw1 = "I thought slide is enough."

    raw1 = """

"""

    print(predict_from_raw_input(raw1))
    # raw2 = "You are awarded a SiPix Digital Camera! call 09061221061 from landline. Delivery within 28days. T Cs Box177. M221BP. 2yr warranty. 150ppm. 16 . p p£3.99"
    # print(predict_from_raw_input(raw2))
