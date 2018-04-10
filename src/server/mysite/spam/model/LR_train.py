import tensorflow as tf
import numpy as np
import tarfile
import os
import matplotlib.pyplot as plt
import time


# DATA INPUT #
def csv_to_numpy_array(filePath, delimiter):
    result = np.genfromtxt(filePath, delimiter=delimiter, dtype=None)
    print(result.shape)
    return result

# Import data of different Type:
# 1: basic Bag of Word features matrix
# 2: Plus Character-level-n-gram features matrix

def import_data(type):
    if "data" not in os.listdir(os.getcwd()):
        raise Exception('data', 'not presented')
    else:
        pass

    if type == 1:
        print("loading training data")
        train_X = csv_to_numpy_array("data/trainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/trainY.csv", delimiter="\t")
        print("loading test data")
        test_X = csv_to_numpy_array("data/testX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/testY.csv", delimiter="\t")
    elif type == 2:
        print("loading training data")
        train_X = csv_to_numpy_array("data/biTrainX.csv", delimiter="\t")
        train_Y = csv_to_numpy_array("data/biTrainY.csv", delimiter="\t")
        print("loading test data")
        test_X = csv_to_numpy_array("data/biTestX.csv", delimiter="\t")
        test_Y = csv_to_numpy_array("data/biTestY.csv", delimiter="\t")
    else:
        raise Exception('data', 'unsupported type')

    return train_X, train_Y, test_X, test_Y


# Global variables use JAVA Camel case,
# Function's instance variables use dash lower case
# Requires Parameter Tunings
if __name__ == '__main__':
    workMode = 2
    trainX, trainY, testX, testY = import_data(workMode)
    # print(trainX.shape)
    # print(trainY.shape)
    # print(testX.shape)
    # print(testY.shape)
    numFeatures = trainX.shape[1]
    numLabels = trainY.shape[1]
    numTrainExamples = trainX.shape[0]
    # print(numFeatures)
    # print(numLabels)
    # print(numTrainExamples)
    numEpochs = 27000

    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trainX.shape[0],
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

    # Operators
    init_OP = tf.global_variables_initializer()
    # Use sigmoid function with FP
    apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
    add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
    activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")
    # Use MSE, i.e. L2 loss function as Cost Function
    cost_OP = tf.nn.l2_loss(activation_OP - tY, name="squared_error_cost")
    # Use Gradient Descent as Optimization Algorithm
    training_OP = tf.train.GradientDescentOptimizer(learningRate).minimize(cost_OP)

    # Run the Training phase
    sess = tf.Session()
    sess.run(init_OP)

    # Add extra Operators for better analysis the training phase
    # argmax(activation_OP, 1) gives the label our model thought was most likely
    # argmax(yGold, 1) is the correct label
    correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(tY, 1))
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    # Three summary operators on OP {activation & accuracy & cost}
    activation_summary_OP = tf.summary.histogram("output", activation_OP)
    accuracy_summary_OP = tf.summary.scalar("accuracy", accuracy_OP)
    cost_summary_OP = tf.summary.scalar("cost", cost_OP)
    # Summary ops to check how variables (W, b) are updating after each iteration
    weightSummary = tf.summary.histogram("weights", weights.eval(session=sess))
    biasSummary = tf.summary.histogram("biases", bias.eval(session=sess))

    # All summary
    all_summary_OPS = tf.summary.merge_all()
    writer = tf.summary.FileWriter("summary_logs", sess.graph)


    # Begin Training Now
    # Initialize reporting variables
    cost = 0
    diff = 1
    epoch_values = []
    accuracy_values = []
    cost_values = []

    # Loop for each Epoch
    for i in range(numEpochs):
        if i > 1 and diff < .0001:
            print("change in cost %g; convergence." % diff)
            break
        else:
            # Run training step
            step = sess.run(training_OP, feed_dict={X: trainX, tY: trainY})
            # Report occasional stats
            if i % 100 == 0:
                # Add epoch to epoch_values
                epoch_values.append(i)
                # Generate accuracy stats on test data
                summary_results, train_accuracy, newCost = sess.run(
                    [all_summary_OPS, accuracy_OP, cost_OP],
                    feed_dict={X: trainX, tY: trainY}
                )
                # Add accuracy to live graphing variable
                accuracy_values.append(train_accuracy)
                # Add cost to live graphing variable
                cost_values.append(newCost)
                # Write summary stats to writer
                writer.add_summary(summary_results, i)
                # Re-assign values for variables
                diff = abs(newCost - cost)
                cost = newCost

                # generate print statements
                print("step %d, training accuracy %g" % (i, train_accuracy))
                print("step %d, cost %g" % (i, newCost))
                print("step %d, change in cost %g" % (i, diff))

                # Plot progress to our two subplots
                # accuracyLine, = ax1.plot(epoch_values, accuracy_values)
                # costLine, = ax2.plot(epoch_values, cost_values)
                # fig.canvas.draw()
                # time.sleep(1)

    print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                          feed_dict={X: testX,
                                                                     tY: testY})))
    saver = tf.train.Saver()
    saver.save(sess, os.getcwd() + "/weights/mode"+str(workMode)+"_trained_variables.ckpt")
    sess.close()





