import os
import pickle
from time import sleep

import tensorflow as tf
from tensorflow.contrib import rnn

from server.mysite.spam.model.gmail.feature_extraction import generate_model_in_memory

# Data cannot be added to IDE scope
# Otherwise the IDE will be too slow
DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'
# Tri-Gram mode
operationalMode = 3









def next_batch(batch_size, batch_id, total_x, total_y):
    begin_index = (batch_id * batch_size) % (total_x.shape[0])
    if begin_index + batch_size < total_x.shape[0]:
        end_index = begin_index + batch_size
    else:
        end_index = total_x.shape[0]
    # test[numpy.logical_or.reduce([test[:, 1] == x for x in wanted])]
    b_x = total_x[begin_index:end_index, :]
    b_y = total_y[begin_index:end_index, :]

    return b_x, b_y


if __name__ == '__main__':



    trainX, trainY, testX, testY = generate_model_in_memory(data_prefix=DATA_PREFIX,
                                                            uni_cutoff=12,
                                                            bi_cutoff=20,
                                                            tri_cutoff=20,
                                                            split=.2,
                                                            spam_ham_ratio=2,
                                                            operational_mode=operationalMode)

    numFeatures = trainX.shape[1]
    numLabels = trainY.shape[1]
    numTrainExamples = trainX.shape[0]

    struct = {'features': numFeatures, 'labels': numLabels, 'examples': numTrainExamples}
    with open('features/structure.pickle', 'wb') as f:
        pickle.dump(struct, f)

    timeSteps = 1
    hiddenUnits = 512
    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trainX.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)
    desiredBatches = 60
    desiredEpochs = 50

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

    # Training Process
    init = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init)

    i = 1
    global_max_test_acc = 0
    saver = tf.train.Saver()

    while i < 1 + desiredBatches * desiredEpochs:
        batch_x, batch_y = next_batch(batch_size=batchSize, batch_id=i, total_x=trainX, total_y=trainY)

        batch_x = batch_x.reshape((batch_x.shape[0], timeSteps, numFeatures))

        sess.run(opt, feed_dict={x: batch_x, y: batch_y})

        if i % desiredBatches == 0:
            # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
            # los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})

            train_acc = sess.run(accuracy, feed_dict={x: trainX.reshape(trainX.shape[0], timeSteps,
                                                                        numFeatures),
                                                      y: trainY})
            test_acc = sess.run(accuracy,
                                feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
                                                            numFeatures),
                                           y: testY})
            print("For epoch ", i // desiredBatches)
            print("Total Accuracy on train set ", train_acc)
            print("Total Accuracy on test set ", test_acc)
            # print("Loss ", los)
            print("__________________")

            if test_acc > global_max_test_acc:
                global_max_test_acc = test_acc
                saver.save(sess,
                           os.getcwd() + "/weights_lstm/lstm_mode" + str(operationalMode) + "_trained_variables.ckpt")
            print('Global Max Test Accuracy:', global_max_test_acc)
            # sleep(3)
        sleep(0.1)
        i = i + 1
    print("final accuracy on test set: %s" % str(sess.run(accuracy,
                                                          feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
                                                                                      numFeatures),
                                                                     y: testY})))
    sess.close()
