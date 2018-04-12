import numpy as np
import tensorflow as tf
import tarfile
import os

###################
### IMPORT DATA ###
###################
from .util import data_helper
# from server.mysite.spam.model.util import data_helper

dataType = 2
trainX, trainY, testX, testY = data_helper.import_data(dataType)
uniFeatureDict, biGramFeatureDict = data_helper.import_features_dict(dataType)
# print("UniFeature size: %s", str(len(uniFeatureDict)))
# print("BiFeature size: %s", str(len(biGramFeatureDict)))

#########################
### GLOBAL PARAMETERS ###
#########################

# Get our dimensions for our different variables and placeholders:
# numFeatures = the number of words extracted from each email
numFeatures = trainX.shape[1]
# numLabels = number of classes we are predicting (here just 2: ham or spam)
numLabels = trainY.shape[1]

# create a tensorflow session
sess = tf.Session()

####################
### PLACEHOLDERS ###
####################

# X = X-matrix / feature-matrix / data-matrix... It's a tensor to hold our email
# data. 'None' here means that we can hold any number of emails
X = tf.placeholder(tf.float32, [None, numFeatures])
# yGold = Y-matrix / label-matrix / labels... This will be our correct answers
# matrix. Every row has either [1,0] for SPAM or [0,1] for HAM. 'None' here
# means that we can hold any number of emails
yGold = tf.placeholder(tf.float32, [None, numLabels])

#################
### VARIABLES ###
#################

# all values must be initialized to a value before loading can occur

weights = tf.Variable(tf.zeros([numFeatures, numLabels]))

bias = tf.Variable(tf.zeros([1, numLabels]))

########################
### OPS / OPERATIONS ###
########################

# since we don't have to train the model, the only Ops are the prediction operations

apply_weights_OP = tf.matmul(X, weights, name="apply_weights")
add_bias_OP = tf.add(apply_weights_OP, bias, name="add_bias")
activation_OP = tf.nn.sigmoid(add_bias_OP, name="activation")

# argmax(activation_OP, 1) gives the label our model thought was most likely
# argmax(yGold, 1) is the correct label
correct_predictions_OP = tf.equal(tf.argmax(activation_OP, 1), tf.argmax(yGold, 1))

# False is 0 and True is 1, what was our average?
accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

# Initializes everything we've defined made above, but doesn't run anything
# until sess.run()
init_OP = tf.global_variables_initializer()

sess.run(init_OP)  # initialize variables BEFORE loading

# load variables from file
saver = tf.train.Saver()
saver.restore(sess, os.getcwd() + "/weights/mode2_trained_variables.ckpt")


#####################
### RUN THE GRAPH ###
#####################

# Initialize all tensorflow objects
# sess.run(init_OP)

# method for converting tensor label to string label
def labelToString(label):
    if np.argmax(label) == 0:
        return "ham"
    else:
        return "spam"


# make prediction on a given test set item
def predict(features, goldLabel):
    # run through graph
    # tensor_prediction = sess.run(activation_OP, feed_dict={X: features.reshape(1, len(features)), yGold: goldLabel.reshape(1, len(goldLabel))})      #had to make sure that each input in feed_dict was an array
    tensor_prediction = sess.run(activation_OP, feed_dict={X: features,
                                                           yGold: goldLabel})  # had to make sure that each input in feed_dict was an array
    prediction = labelToString(tensor_prediction)
    actual = labelToString(goldLabel)
    print("regression predicts email to be %s and is actually %s" % (prediction, actual))

# Two functions used for server
def predict_from_raw_input(raw_input):
    uniMatrix = data_helper.generate_sample_unigram(raw_input, uniFeatureDict)
    biMatrix = data_helper.generate_sample_ngram(raw_input, biGramFeatureDict)
    combinedMatrixX = np.concatenate((uniMatrix, biMatrix), axis=1)
    tensor_prediction = sess.run(activation_OP, feed_dict={X: combinedMatrixX})  # had to make sure that each input in feed_dict was an array
    print(tensor_prediction)
    prediction = single_label_to_string(tensor_prediction)
    return prediction

def single_label_to_string(tensor_prediction):
    # 2nd vector value > 1st vector value => [0,1] => 'ham'
    if tensor_prediction[0][1] > tensor_prediction[0][0]:
        return 'ham'
    else:
        return 'spam'




def predict_all(vectorX, vectorY):
    prediction, evaluation = sess.run([activation_OP, accuracy_OP], feed_dict={X: testX, yGold: testY})
    for i in range(len(testX)):
        print("regression predicts email %s to be %s and is actually %s" % (
        str(i + 1), labelToString(prediction[i]), labelToString(testY[i])))
    print("overall accuracy of dataset: %s percent" % str(evaluation))
    return prediction


if __name__ == "__main__":

    # show predictions and accuracy of entire test set
    # prediction, evaluation = sess.run([activation_OP, accuracy_OP], feed_dict={X: testX, yGold: testY})

    # for i in range(len(testX)):
    #     print("regression predicts email %s to be %s and is actually %s" % (
    #     str(i + 1), labelToString(prediction[i]), labelToString(testY[i])))
    # print("overall accuracy of dataset: %s percent" % str(evaluation))

    raw = "I thought slide is enough."
    predict_from_raw_input(raw)



