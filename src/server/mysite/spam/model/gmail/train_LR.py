import os
import pickle
from time import sleep

import numpy as np
import tensorflow as tf

# Global variables use JAVA Camel case,
# Function's instance variables use dash lower case
# Requires Parameter Tunings
from server.mysite.spam.model.gmail.util import FeatureExtraction

DATA_PREFIX = '/Users/jianyuzuo/Workspaces/CSCE665_project/'
DATA_PREFIX_WIN = 'D:\\workplaces\\665\\'

# Tri-Gram mode
operationalMode = 3
# Small Medium Large FeatureSets
CUTOFF_SETTINGS = [(18, 28, 26), (12, 22, 20), (8, 16, 14)]
CUTOFF_STRINGS = ['small', 'medium', 'large']
# Spam Ham Ratio 1,2,3
SPAM_HAM_RATIOS = [1, 2, 3]


def model_tuning_single(epochs=20000, is_win_platform=False):
    print('Tuning LR model')
    if is_win_platform:
        data_prefix = DATA_PREFIX_WIN
        weights_output_prefix = os.getcwd() + "\\weights_lr\\"
    else:
        data_prefix = DATA_PREFIX
        weights_output_prefix = os.getcwd() + "/weights_lr/"
    for j in range(0, 1):
        (uni, bi, tri) = CUTOFF_SETTINGS[j]
        print('Training on', CUTOFF_STRINGS[j], 'scale')
        for k in range(2, 3):
            ratio = SPAM_HAM_RATIOS[k]
            print('Training on spam ratio', ratio)
            pickle_name_prefix = 'lr_operationMode-' + str(operationalMode) \
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
            print('Features for training:', numFeatures)
            structure = {'features': numFeatures, 'labels': numLabels, 'examples': numTrainExamples}

            if is_win_platform:
                feature_prefix = 'features\\lr_operationMode-' + str(operationalMode) + '_structure_'
            else:
                feature_prefix = 'features/lr_operationMode-' + str(operationalMode) + '_structure_'
            with open(feature_prefix + CUTOFF_STRINGS[j] + '_' + str(ratio) + '.pickle', 'wb') as f:
                pickle.dump(structure, f)

            desiredEpochs = epochs
            learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                                      global_step=1,
                                                      decay_steps=numTrainExamples,
                                                      decay_rate=0.95,
                                                      staircase=True)

            # X ~ N*M,
            # N: # of examples, M:# of features
            X = tf.placeholder(tf.float32, [None, numFeatures])
            # tY ~ N*C, tY short for "true Y"
            # C: # of classes
            tY = tf.placeholder(tf.float32, [None, numLabels])

            # Initialization weights [Adam or other initializations may be tried]
            weights = tf.Variable(tf.random_normal([numFeatures, numLabels],
                                                   mean=0,
                                                   stddev=(np.sqrt(6 / numFeatures +
                                                                   numLabels + 1)),
                                                   name="weights"))
            bias = tf.Variable(tf.random_normal([1, numLabels],
                                                mean=0,
                                                stddev=(np.sqrt(6 / numFeatures + numLabels + 1)),
                                                name="bias"))

            # Use sigmoid function with FP
            apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
            add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
            activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
            # Use MSE, i.e. L2 loss function as Cost Function
            cost_OP = tf.nn.l2_loss(activation_OP - tY, name="squared_error_cost")
            # Use Gradient Descent as Optimization Algorithm
            training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

            # Add extra Operators for better analysis the training phase
            # ARGMAX 0 => Ham 1 => Spam
            predicted_OP = tf.argmax(activation_OP, 1)
            truth_OP = tf.argmax(tY, 1)
            true_positive_OP = tf.count_nonzero(predicted_OP * truth_OP)
            true_negative_OP = tf.count_nonzero((predicted_OP - 1) * (truth_OP - 1))
            false_positive_OP = tf.count_nonzero(predicted_OP * (truth_OP - 1))
            false_negative_OP = tf.count_nonzero((predicted_OP - 1) * truth_OP)
            precision_OP = true_positive_OP / (true_positive_OP + false_positive_OP)
            recall_OP = true_positive_OP / (true_positive_OP + false_negative_OP)
            f1_OP = 2 * precision_OP * recall_OP / (precision_OP + recall_OP)
            accuracy_OP = (true_positive_OP + true_negative_OP) / \
                          (true_positive_OP + true_negative_OP + false_positive_OP + false_negative_OP)

            # Run the Training phase
            init_OP = tf.global_variables_initializer()
            sess = tf.Session()
            sess.run(init_OP)
            saver = tf.train.Saver()

            # TODO: Figure out how to use tf.summary package
            # activation_summary_OP = tf.summary.histogram("output", activation_OP)
            # accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
            # cost_summary_OP = tf.summary.scalar("cost", cost_OP)
            # # Summary ops to check how variables (W, b) are updating after each iteration
            # weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
            # biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

            cost = 0
            global_max_f1 = 0
            for i in range(desiredEpochs + 1):
                sess.run(training_OP, feed_dict={X: trainX, tY: trainY})
                if i % 50 == 0:
                    train_acc, train_pc, train_rc, train_f1, train_cost = sess.run(
                        [accuracy_OP, precision_OP, recall_OP, f1_OP, cost_OP],
                        feed_dict={X: trainX, tY: trainY})
                    test_acc, test_pc, test_rc, test_f1 = sess.run([accuracy_OP, precision_OP, recall_OP, f1_OP],
                                                                   feed_dict={X: testX, tY: testY})
                    print("For epoch ", i)
                    # print("Train[ACC, PC, RC]:\t ", train_acc, train_pc, train_rc)
                    print("Train[F1]:\t\t\t ", train_f1)
                    # print("Test[ACC, PC, RC]:\t ", test_acc, test_pc, test_rc)
                    print("Test[F1]:\t\t\t ", test_f1)
                    print("______________________________________")
                    if test_f1 >= global_max_f1:
                        optimal_parameters['acc'] = test_acc
                        optimal_parameters['pc'] = test_pc
                        optimal_parameters['rc'] = test_rc
                        optimal_parameters['f1'] = test_f1
                        optimal_parameters['step'] = i
                        if test_f1 > global_max_f1:
                            saver.save(sess,
                                       weights_output_prefix + "lr_operationMode-" + str(
                                           operationalMode) + '_cutoff-' +
                                       CUTOFF_STRINGS[j] + '_ratio-' + str(ratio) + "_trained_variables.ckpt")
                            with open(weights_output_prefix + pickle_name_prefix + 'optimalParameters.pickle',
                                      'wb') as f:
                                pickle.dump(optimal_parameters, f)
                            print('Successfully saved new weights')
                        global_max_f1 = test_f1
                    print('Global Max F1:', global_max_f1, 'on step:', optimal_parameters['step'])
                    diff = abs(train_cost - cost)
                    cost = train_cost
                    if i > 1 and diff < .0000001:
                        print("change in cost %g; convergence." % diff)
                        break

            print("final optimal parameters: ",
                  str(optimal_parameters['acc']), str(optimal_parameters['pc']),
                  str(optimal_parameters['rc']), str(optimal_parameters['f1']))
            with open(weights_output_prefix + pickle_name_prefix + 'optimalParameters.pickle', 'wb') as f:
                pickle.dump(optimal_parameters, f)
            sess.close()

        sleep(10)
    ################################################################################
    pass


if __name__ == '__main__':
    model_tuning_single(28000, True)
