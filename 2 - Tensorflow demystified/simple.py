# Copyright 2019 Avinash Meetoo
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Tensorflow demystified
# https://chatbotslife.com/tensorflow-demystified-80987184faf7

# There’s no black-magic here: math is math.

import numpy as np
import random
import tensorflow as tf

def create_feature_sets_and_labels(test_size = 0.3):
    # known patterns (5 features) output of [1] of positions [0, 4] == 1
    features = []
    features.append([[0, 0, 0, 0, 0], [0,1]])
    features.append([[0, 0, 0, 0, 1], [0,1]])
    features.append([[0, 0, 0, 1, 1], [0,1]])
    features.append([[0, 0, 1, 1, 1], [0,1]])
    features.append([[0, 1, 1, 1, 1], [0,1]])
    features.append([[1, 1, 1, 1, 0], [0,1]])
    features.append([[1, 1, 1, 0, 0], [0,1]])
    features.append([[1, 1, 0, 0, 0], [0,1]])
    features.append([[1, 0, 0, 0, 0], [0,1]])
    features.append([[1, 0, 0, 1, 0], [0,1]])
    features.append([[1, 0, 1, 1, 0], [0,1]])
    features.append([[1, 1, 0, 1, 0], [0,1]])
    features.append([[0, 1, 0, 1, 1], [0,1]])
    features.append([[0, 0, 1, 0, 1], [0,1]])
    features.append([[1, 0, 1, 1, 1], [1,0]])
    features.append([[1, 1, 0, 1, 1], [1,0]])
    features.append([[1, 0, 1, 0, 1], [1,0]])
    features.append([[1, 0, 0, 0, 1], [1,0]])
    features.append([[1, 1, 0, 0, 1], [1,0]])
    features.append([[1, 1, 1, 0, 1], [1,0]])
    features.append([[1, 1, 1, 1, 1], [1,0]])
    features.append([[1, 0, 0, 1, 1], [1,0]])

    # shuffle out features and turn into np.array
    random.shuffle(features)
    features = np.array(features)

    # split a portion of the features into tests
    testing_size = int(test_size*len(features))

    # create train and test lists
    train_x = list(features[:,0][:-testing_size])
    train_y = list(features[:,1][:-testing_size])
    test_x = list(features[:,0][-testing_size:])
    test_y = list(features[:,1][-testing_size:])

    # notice in the above setup we are shuffling the data (‘features’)
    # and using 2/3 of it for training, 1/3 of it for testing.
    # The ratio is the parameter ‘test_size’. 
    return train_x, train_y, test_x, test_y

train_x, train_y, test_x, test_y = create_feature_sets_and_labels()

# hidden layers and their nodes
n_nodes_hl1 = 32
n_nodes_hl2 = 32

# classes in our output
n_classes = 2
# iterations and batch-size to build out model
hm_epochs = 1000
batch_size = 4
    
x = tf.placeholder('float')
y = tf.placeholder('float')

# random weights and bias for our layers

# Our data is loaded, we’ll use 20 nodes in 2 hidden layers
# and initialize weights and biases with random values.
# We also define our output layer.

hidden_1_layer = { 'f_fum':  n_nodes_hl1,
                   'weight': tf.Variable(tf.random_normal([len(train_x[0]), n_nodes_hl1])),
                   'bias':   tf.Variable(tf.random_normal([n_nodes_hl1])) }

hidden_2_layer = { 'f_fum':  n_nodes_hl2,
                   'weight': tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                   'bias':   tf.Variable(tf.random_normal([n_nodes_hl2])) }

output_layer = { 'f_fum':  None,
                 'weight': tf.Variable(tf.random_normal([n_nodes_hl2, n_classes])),
                 'bias':   tf.Variable(tf.random_normal([n_classes])) }

# Let's define the mathematical equations for our model
# our predictive model's definition
def neural_network_model(data):
    # hidden layer 1: (data * W) + b
    l1 = tf.add(tf.matmul(data, hidden_1_layer['weight']), hidden_1_layer['bias'])
    l1 = tf.sigmoid(l1)

    # hidden layer 2: (hidden_layer_1 * W) + b
    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weight']), hidden_2_layer['bias'])
    l2 = tf.sigmoid(l2)

    # output: (hidden_layer_2 * W) + b
    output = tf.matmul(l2, output_layer['weight']) + output_layer['bias']

    return output

# training our model
def train_neural_network(x):
    # use the model definition
    prediction = neural_network_model(x)

    # formula for cost (error)
    # the softmax function is a function that takes as input a vector of K real numbers,
    # and normalizes it into a probability distribution consisting of K probabilities.
    # That is, prior to applying softmax, some vector components could be negative,
    # or greater than one; and might not sum to 1; but after applying softmax,
    # each component will be in the interval (0,1), and the components will add up to 1
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits = prediction, labels = y))
    
    # optimize for cost using GradientDescent
    # Gradient descent is a first-order iterative optimization algorithm
    # for finding the minimum of a function.
    optimizer = tf.train.GradientDescentOptimizer(1).minimize(cost)

    # Tensorflow session
    with tf.Session() as sess:
        # initialize our variables
        sess.run(tf.global_variables_initializer())

        # loop through specified number of iterations
        for epoch in range(hm_epochs):
            epoch_loss = 0
            i = 0
            # handle batch sized chunks of training data
            while i < len(train_x):
                start = i
                end = i + batch_size
                batch_x = np.array(train_x[start:end])
                batch_y = np.array(train_y[start:end])

                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x, y: batch_y})
                epoch_loss += c
                i += batch_size
                last_cost = c

            # print cost updates along the way
            if (epoch % (hm_epochs / 5)) == 0:
                print('Epoch', epoch, 'completed out of',hm_epochs,'cost:', last_cost)
        
        # print accuracy of our model
        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy:',accuracy.eval({x: test_x, y: test_y}))

        # print predictions using our model
        for i, t in enumerate(test_x):
            print ('prediction for:', test_x[i])
            output = prediction.eval(feed_dict = {x: [test_x[i]]})
            # normalize the prediction values
            print(tf.sigmoid(output[0][0]).eval(), tf.sigmoid(output[0][1]).eval())
        
train_neural_network(x)
