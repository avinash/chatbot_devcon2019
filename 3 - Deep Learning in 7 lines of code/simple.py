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

# Deep Learning in 7 lines of code
# https://chatbotslife.com/deep-learning-in-7-lines-of-code-7879a8ef8cfb

# TFLearn: Deep learning library featuring a higher-level API for TensorFlow.

import random
import numpy as np
import tflearn

# Build neural network
net = tflearn.input_data(shape=[None, 5])
net = tflearn.fully_connected(net, 32)
net = tflearn.fully_connected(net, 32)

# the softmax function is a function that takes as input a vector of K real numbers,
# and normalizes it into a probability distribution consisting of K probabilities.
# That is, prior to applying softmax, some vector components could be negative,
# or greater than one; and might not sum to 1; but after applying softmax,
# each component will be in the interval (0,1), and the components will add up to 1
net = tflearn.fully_connected(net, 2, activation='softmax')

# What is Linear Regression?  It’s a Supervised Learning algorithm which goal
# is to predict continuous, numerical values based on given data input.
# From the geometrical perspective, each data sample is a point.
# Linear Regression tries to find parameters of the linear function,
# so the distance between the all the points and the line is as small as possible.
# Algorithm used for parameters update is called Gradient Descent.
net = tflearn.regression(net)

# Define model and setup tensorboard
# DNN means Deep Neural Network
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# Known patterns (5 features) output of [1] of positions [0, 4] == 1
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

# Notice in the above setup we are shuffling the data (‘features’)
# and using 2/3 of it for training, 1/3 of it for testing.
# The ratio is the parameter ‘test_size’. 
test_size = 0.3

# split a portion of the features into tests
testing_size = int(test_size*len(features))

# create train and test lists
train_x = list(features[:,0][:-testing_size])
train_y = list(features[:,1][:-testing_size])
test_x = list(features[:,0][-testing_size:])
test_y = list(features[:,1][-testing_size:])

# Start training (apply gradient descent algorithm)
# Gradient descent is a first-order iterative optimization algorithm
# for finding the minimum of a function.
model.fit(train_x, train_y, n_epoch=1000, batch_size=16, show_metric=True)

# print predictions using our model
for i, t in enumerate(test_x):
    print('prediction for:', test_x[i])
    print(model.predict([test_x[i]]))
