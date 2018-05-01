import os
import pickle
from time import sleep

import tensorflow as tf
from tensorflow.contrib import rnn

# Data cannot be added to IDE scope
# Otherwise the IDE will be too slow

# from . import feature_extraction
from server.mysite.spam.model.gmail.util import FeatureExtraction

DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'
DATA_PREFIX_WIN = 'D:\\workplaces\\665\\'

# Tri-Gram mode
operationalMode = 1
# Small Medium Large FeatureSets
CUTOFF_SETTINGS = [(18, 28, 26), (12, 22, 20), (8, 16, 14)]
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
                        epochs=100,
                        is_win_platform=False):
    if is_win_platform:
        data_prefix = DATA_PREFIX_WIN
        weights_output_prefix = os.getcwd() + "\\weights_lstm\\"
    else:
        data_prefix = DATA_PREFIX
        weights_output_prefix = os.getcwd() + "/weights_lstm/"
    print('Training with hidden units', hidden_units)
    for j in range(1, 2):
        (uni, bi, tri) = CUTOFF_SETTINGS[j]
        print('Training on', CUTOFF_STRINGS[j], 'scale')
        for k in range(2, 3):
            ratio = SPAM_HAM_RATIOS[k]
            print('Training on spam ratio', ratio)
            pickle_name_prefix = 'operationMode-' + str(operationalMode) \
                                 + '_cutoff-' + CUTOFF_STRINGS[j] \
                                 + '_ratio-' + str(ratio)
            # Had to run this command before building the graph
            tf.reset_default_graph()
            # Dict used to save optimal params
            optimal_parameters = {}
            trainX, trainY, testX, testY = FeatureExtraction.generate_model_in_memory(data_prefix=data_prefix,
                                                                                      is_win_platform=is_win_platform,
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

            if is_win_platform:
                feature_prefix = 'features\\structure_'
            else:
                feature_prefix = 'features/structure_'
            with open(feature_prefix + CUTOFF_STRINGS[j] + '_' + str(ratio) + '.pickle', 'wb') as f:
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
            cost = 0
            global_max_f1 = 0
            saver = tf.train.Saver()

            while i < 1 + desiredBatches * desiredEpochs:
                batch_x, batch_y = next_batch(batch_size=batchSize, batch_id=i, total_x=trainX, total_y=trainY)

                batch_x = batch_x.reshape((batch_x.shape[0], timeSteps, numFeatures))

                sess.run(opt_OP, feed_dict={x: batch_x, y: batch_y})

                if i % desiredBatches == 0:

                    train_acc, train_pc, train_rc, train_f1, train_cost = sess.run(
                        [accuracy_OP, precision_OP, recall_OP, f1_OP, loss_OP],
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

                    if test_f1 >= global_max_f1:
                        optimal_parameters['acc'] = test_acc
                        optimal_parameters['pc'] = test_pc
                        optimal_parameters['rc'] = test_rc
                        optimal_parameters['f1'] = test_f1
                        optimal_parameters['step'] = i
                        if test_f1 > global_max_f1:
                            saver.save(sess,
                                       weights_output_prefix + "lstm_operationMode-" + str(
                                           operationalMode) + '_cutoff-' +
                                       CUTOFF_STRINGS[j] + '_ratio-' + str(ratio) + "_trained_variables.ckpt")
                            with open(weights_output_prefix + pickle_name_prefix + 'optimalParameters.pickle',
                                      'wb') as f:
                                pickle.dump(optimal_parameters, f)
                            print('Successfully saved new weights')
                        global_max_f1 = test_f1
                    print('Global Max F1:', global_max_f1, 'on step:', optimal_parameters['step'] // desiredBatches)
                    diff = abs(train_cost - cost)
                    cost = train_cost
                    if i > 1 and diff < .0000001:
                        print("change in cost %g; convergence." % diff)
                        break
                    # sleep(3)
                # sleep(0.1)
                i = i + 1
            print("final optimal parameters: ",
                  str(optimal_parameters['acc']), str(optimal_parameters['pc']),
                  str(optimal_parameters['rc']), str(optimal_parameters['f1']))
            with open(weights_output_prefix + pickle_name_prefix + 'optimalParameters.pickle', 'wb') as f:
                pickle.dump(optimal_parameters, f)
            sess.close()

        sleep(300)
    pass


if __name__ == '__main__':
    # model_tuning_single(hidden_units=128, epochs=1000, is_win_platform=True)
    model_tuning_single(hidden_units=256, epochs=1000, is_win_platform=True)
    # model_tuning_single(hidden_units=512, epochs=1000)

    # print(os.getcwd())
    # print('End')
    pass
