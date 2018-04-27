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
# Small Medium Large FeatureSets
CUTOFF_SETTINGS = [(15, 25, 25), (12, 20, 20), (8, 15, 18)]
CUTOFF_STRINGS = ['small', 'medium', 'large']
# Spam Ham Ratio 1,2,3
SPAM_HAM_RATIOS = [1, 2, 3]


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


# A combined method for tuning a single model without cross-validation
def model_tuning_single(hidden_units=256,
                        batches=50,
                        epochs=100):
    for j in range(3):
        (uni, bi, tri) = CUTOFF_SETTINGS[j]
        for k in range(3):
            ratio = SPAM_HAM_RATIOS[k]
            pickle_name_prefix = 'operationMode-' + str(operationalMode) \
                                 + '_cutoff-' + CUTOFF_STRINGS[j] \
                                 + '_ratio-' + str(ratio)

            trainX, trainY, testX, testY = generate_model_in_memory(data_prefix=DATA_PREFIX,
                                                                    uni_cutoff=uni,
                                                                    bi_cutoff=bi,
                                                                    tri_cutoff=tri,
                                                                    split=.2,
                                                                    spam_ham_ratio=ratio,
                                                                    operational_mode=operationalMode,
                                                                    file_prefix=pickle_name_prefix)
            numFeatures = trainX.shape[1]
            numLabels = trainY.shape[1]
            numTrainExamples = trainX.shape[0]
            print('Features:', numFeatures)
            struct = {'features': numFeatures, 'labels': numLabels, 'examples': numTrainExamples}
            with open('features/structure_' + CUTOFF_STRINGS[j] + '_' + str(ratio) + '.pickle', 'wb') as f:
                pickle.dump(struct, f)

            timeSteps = 1
            hiddenUnits = hidden_units
            learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                                      global_step=1,
                                                      decay_steps=trainX.shape[0],
                                                      decay_rate=0.95,
                                                      staircase=True)
            desiredBatches = batches
            desiredEpochs = epochs
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

            inputs = tf.unstack(x, timeSteps, 1)
            lstm_layer = rnn.BasicLSTMCell(hiddenUnits, forget_bias=1)
            outputs, _ = rnn.static_rnn(lstm_layer, inputs, dtype="float32")

            # converting last output of dimension [batch_size,num_units]
            # to [batch_size,n_classes] by out_weight multiplication
            prediction_OP = tf.matmul(outputs[-1], outWeights) + outBias
            # loss_function
            loss_OP = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_OP, labels=y))
            # optimization *** CORE
            opt_OP = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss_OP)

            # ARGMAX 0 => Ham 1 => Spam
            # model evaluation
            # correct_prediction = tf.equal(tf.argmax(prediction_OP, 1), tf.argmax(y, 1))
            # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

            predicted_OP = tf.argmax(prediction_OP, 1)
            truth_OP = tf.argmax(y, 1)

            true_positive_OP = tf.count_nonzero(predicted_OP * truth_OP)
            true_negative_OP = tf.count_nonzero((predicted_OP - 1) * (truth_OP - 1))
            false_positive_OP = tf.count_nonzero(predicted_OP * (truth_OP - 1))
            false_negative_OP = tf.count_nonzero((predicted_OP - 1) * truth_OP)

            precision_OP = true_positive_OP / (true_positive_OP + false_positive_OP)
            recall_OP = true_positive_OP / (true_positive_OP + false_negative_OP)
            f1_OP = 2 * precision_OP * recall_OP / (precision_OP + recall_OP)
            accuracy_OP = (true_positive_OP + true_negative_OP) / \
                          (true_positive_OP + true_negative_OP + false_positive_OP + false_negative_OP)

            # Training Process
            init = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init)

            i = 1
            global_max_f1 = 0
            saver = tf.train.Saver()

            while i < 1 + desiredBatches * desiredEpochs:
                batch_x, batch_y = next_batch(batch_size=batchSize, batch_id=i, total_x=trainX, total_y=trainY)

                batch_x = batch_x.reshape((batch_x.shape[0], timeSteps, numFeatures))

                sess.run(opt_OP, feed_dict={x: batch_x, y: batch_y})

                if i % desiredBatches == 0:

                    train_acc, train_pc, train_rc, train_f1 = sess.run([accuracy_OP, precision_OP, recall_OP, f1_OP],
                                                                       feed_dict={
                                                                           x: trainX.reshape(trainX.shape[0], timeSteps,
                                                                                             numFeatures),
                                                                           y: trainY})
                    test_acc, test_pc, test_rc, test_f1 = sess.run([accuracy_OP, precision_OP, recall_OP, f1_OP],
                                                                   feed_dict={
                                                                       x: testX.reshape(testX.shape[0], timeSteps,
                                                                                        numFeatures),
                                                                       y: testY})
                    print("For epoch ", i // desiredBatches)
                    print("Train[ACC, PC, RC]:\t ", train_acc, train_pc, train_rc)
                    print("Train[F1]:\t\t\t ", train_f1)
                    print("Test[ACC, PC, RC]:\t ", test_acc, test_pc, test_rc)
                    print("Test[F1]:\t\t\t ", test_f1)
                    # print("Loss ", los)
                    print("__________________")

                    if test_f1 > global_max_f1:
                        global_max_f1 = test_f1
                        saver.save(sess,
                                   os.getcwd() + "/weights_lstm/lstm_operationMode-" + str(
                                       operationalMode) + '_cutoff-' +
                                   CUTOFF_STRINGS[j] + '_ratio-' + str(ratio) + "_trained_variables.ckpt")
                    print('Global Max F1:', global_max_f1)
                    # sleep(3)
                sleep(0.1)
                i = i + 1
            print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                                  feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
                                                                                              numFeatures),
                                                                             y: testY})))
            sess.close()

    pass


if __name__ == '__main__':
    model_tuning_single(hidden_units=128)

    # trainX, trainY, testX, testY = generate_model_in_memory(data_prefix=DATA_PREFIX,
    #                                                         uni_cutoff=12,
    #                                                         bi_cutoff=20,
    #                                                         tri_cutoff=20,
    #                                                         split=.2,
    #                                                         spam_ham_ratio=2,
    #                                                         operational_mode=operationalMode)
    #
    # numFeatures = trainX.shape[1]
    # numLabels = trainY.shape[1]
    # numTrainExamples = trainX.shape[0]
    #
    # struct = {'features': numFeatures, 'labels': numLabels, 'examples': numTrainExamples}
    # with open('features/structure.pickle', 'wb') as f:
    #     pickle.dump(struct, f)
    #
    # timeSteps = 1
    # hiddenUnits = 200
    # learningRate = tf.train.exponential_decay(learning_rate=0.0008,
    #                                           global_step=1,
    #                                           decay_steps=trainX.shape[0],
    #                                           decay_rate=0.95,
    #                                           staircase=True)
    # desiredBatches = 60
    # desiredEpochs = 50
    # batchSize = numTrainExamples // desiredBatches
    #
    # # weights biases
    # outWeights = tf.Variable(tf.random_normal([hiddenUnits, numLabels]))
    # outBias = tf.Variable(tf.random_normal([numLabels]))
    #
    # # X ~ N*M,
    # # N: # of examples, M:# of features
    # x = tf.placeholder(tf.float32, [None, timeSteps, numFeatures])
    # # tY ~ N*C, tY short for "true Y"
    # # C: # of classes
    # y = tf.placeholder(tf.float32, [None, numLabels])
    #
    # inputs = tf.unstack(x, timeSteps, 1)
    # lstm_layer = rnn.BasicLSTMCell(hiddenUnits, forget_bias=1)
    # outputs, _ = rnn.static_rnn(lstm_layer, inputs, dtype="float32")
    #
    # # converting last output of dimension [batch_size,num_units] to [batch_size,n_classes] by out_weight multiplication
    # prediction_OP = tf.matmul(outputs[-1], outWeights) + outBias
    # # loss_function
    # loss_OP = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction_OP, labels=y))
    # # optimization *** CORE
    # opt_OP = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(loss_OP)
    #
    # # ARGMAX 0 => Ham 1 => Spam
    # # model evaluation
    # # correct_prediction = tf.equal(tf.argmax(prediction_OP, 1), tf.argmax(y, 1))
    # # accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
    #
    # predicted_OP = tf.argmax(prediction_OP, 1)
    # truth_OP = tf.argmax(y, 1)
    #
    # true_positive_OP = tf.count_nonzero(predicted_OP * truth_OP)
    # true_negative_OP = tf.count_nonzero((predicted_OP - 1) * (truth_OP - 1))
    # false_positive_OP = tf.count_nonzero(predicted_OP * (truth_OP - 1))
    # false_negative_OP = tf.count_nonzero((predicted_OP - 1) * truth_OP)
    #
    # precision_OP = true_positive_OP / (true_positive_OP + false_positive_OP)
    # recall_OP = true_positive_OP / (true_positive_OP + false_negative_OP)
    # f1_OP = 2 * precision_OP * recall_OP / (precision_OP + recall_OP)
    # accuracy_OP = (true_positive_OP + true_negative_OP) / \
    #               (true_positive_OP + true_negative_OP + false_positive_OP + false_negative_OP)
    #
    # # Training Process
    # init = tf.global_variables_initializer()
    # sess = tf.Session()
    # sess.run(init)
    #
    # i = 1
    # global_max_f1 = 0
    # saver = tf.train.Saver()
    #
    # while i < 1 + desiredBatches * desiredEpochs:
    #     batch_x, batch_y = next_batch(batch_size=batchSize, batch_id=i, total_x=trainX, total_y=trainY)
    #
    #     batch_x = batch_x.reshape((batch_x.shape[0], timeSteps, numFeatures))
    #
    #     sess.run(opt_OP, feed_dict={x: batch_x, y: batch_y})
    #
    #     if i % desiredBatches == 0:
    #         # acc = sess.run(accuracy, feed_dict={x: batch_x, y: batch_y})
    #         # los = sess.run(loss, feed_dict={x: batch_x, y: batch_y})
    #
    #         train_acc, train_pc, train_rc, train_f1 = sess.run([accuracy_OP, precision_OP, recall_OP, f1_OP],
    #                                                            feed_dict={x: trainX.reshape(trainX.shape[0], timeSteps,
    #                                                                                         numFeatures),
    #                                                                       y: trainY})
    #         test_acc, test_pc, test_rc, test_f1 = sess.run([accuracy_OP, precision_OP, recall_OP, f1_OP],
    #                                                        feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
    #                                                                                    numFeatures),
    #                                                                   y: testY})
    #         print("For epoch ", i // desiredBatches)
    #         print("Train[ACC, PC, RC]:\t ", train_acc, train_pc, train_rc)
    #         print("Train[F1]:\t\t\t ", train_f1)
    #         print("Test[ACC, PC, RC]:\t ", test_acc, test_pc, test_rc)
    #         print("Test[F1]:\t\t\t ", test_f1)
    #         # print("Loss ", los)
    #         print("__________________")
    #
    #         if test_f1 > global_max_f1:
    #             global_max_f1 = test_f1
    #             saver.save(sess,
    #                        os.getcwd() + "/weights_lstm/lstm_mode" + str(operationalMode) + "_trained_variables.ckpt")
    #         print('Global Max F1:', global_max_f1)
    #         # sleep(3)
    #     sleep(0.1)
    #     i = i + 1
    # print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
    #                                                       feed_dict={x: testX.reshape(testX.shape[0], timeSteps,
    #                                                                                   numFeatures),
    #                                                                  y: testY})))
    # sess.close()

    pass
