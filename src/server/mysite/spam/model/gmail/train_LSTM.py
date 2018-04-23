import os
import pickle

import tensorflow as tf
from tensorflow.contrib import rnn

from server.mysite.spam.model.gmail.feature_extraction import generate_model_in_memory

# Data cannot be added to IDE scope
# Otherwise the whole will be too slow
DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'


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


if __name__ == '__main__':
    operationalMode = 3

    trainX, trainY, testX, testY = generate_model_in_memory(DATA_PREFIX, 12, 20, 20, .2, 3, operationalMode)

    numFeatures = trainX.shape[1]
    numLabels = trainY.shape[1]
    numTrainExamples = trainX.shape[0]

    struct = {'features': numFeatures, 'labels': numLabels, 'examples': numTrainExamples}
    with open('features/structure.pickle', 'wb') as f:
        pickle.dump(struct, f)

    timeSteps = 1
    hiddenUnits = 256
    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trainX.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)
    batchSize = numTrainExamples // 10

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

    # Training Process
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    i = 1
    while i < 100:
        batch_x, batch_y = next_batch(batch_size=batchSize, batch_id=i, X=trainX, Y=trainY)

        batch_x = batch_x.reshape((batch_x.shape[0], timeSteps, numFeatures))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if i % 10 == 0:
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})

            total_acc = sess.run(accuracy, feed_dict={x: trainX.reshape(trainX.shape[0], timeSteps,
                                                                        numFeatures),
                                                      y: trainY})
            print("For iter ", i)
            print("Total Accuracy ", total_acc)
            # print("Loss ", los)
            print("__________________")

        i = i + 1
    print("final accuracy on test set: %s" % str(sess.run(accuracy,
                                                          feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
                                                                                      numFeatures),
                                                                     y: testY})))
    saver = tf.train.Saver()
    saver.save(sess, os.getcwd() + "/weights_lstm/lstm_mode" + str(operationalMode) + "_trained_variables.ckpt")
    sess.close()
