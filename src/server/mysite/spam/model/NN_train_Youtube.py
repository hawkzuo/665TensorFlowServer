import tensorflow as tf

from server.mysite.spam.model.util import data_helper


dataMode = 2
trainX, trainY, testX, testY = data_helper.import_data_youtube(dataMode)
numFeatures = trainX.shape[1]
numLabels = trainY.shape[1]
numTrainExamples = trainX.shape[0]

n_nodes_hl1 = 200
n_nodes_hl2 = 200
n_nodes_hl3 = 200

n_classes = numLabels
batch_size = 100

x = tf.placeholder(tf.float32, [None, numFeatures])
y = tf.placeholder(tf.float32, [None, numLabels])


def neural_network_model(data):
    hidden_1_layer = {'weights': tf.Variable(tf.random_normal([numFeatures, n_nodes_hl1])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl1]))}

    hidden_2_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                      'biases': tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights': tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes])), }

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output


def train_neural_network(x):
    prediction = neural_network_model(x)
    learningRate = tf.train.exponential_decay(learning_rate=0.0008,
                                              global_step=1,
                                              decay_steps=trainX.shape[0],
                                              decay_rate=0.95,
                                              staircase=True)
    # OLD VERSION:
    # cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(prediction,y) )
    # NEW:
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer(learning_rate=learningRate).minimize(cost)

    # Extra OPs
    correct_predictions_OP = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    accuracy_OP = tf.reduce_mean(tf.cast(correct_predictions_OP, "float"))

    hm_epochs = 56

    init_OP = tf.global_variables_initializer()
    sess = tf.Session()
    sess.run(init_OP)

    for epoch in range(hm_epochs):
        epoch_accuracy = 0
        epoch_cost = 1000000000000
        for batch_id in range(int(numTrainExamples / batch_size)):
            epoch_x, epoch_y = next_batch(batch_size, batch_id, trainX, trainY)
            _, train_accuracy, c = sess.run([optimizer, accuracy_OP, cost], feed_dict={x: epoch_x, y: epoch_y})

            epoch_accuracy = max(epoch_accuracy, train_accuracy)
            epoch_cost = min(epoch_cost, c)
        if epoch % 2 == 0:
            print('Epoch', epoch, 'Accuracy', epoch_accuracy,'cost:', epoch_cost )

    # correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
    #
    # accuracy = tf.reduce_mean(tf.cast(correct, 'float'))

    print("final accuracy on test set: %s" % str(sess.run(accuracy_OP,
                                                          feed_dict={x: testX,
                                                                     y: testY})))

    sess.close()


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
    train_neural_network(x)
