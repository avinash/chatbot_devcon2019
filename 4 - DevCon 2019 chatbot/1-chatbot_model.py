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

# Contextual Chatbots with Tensorflow
# https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

# Natural Language Processing
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# organize our documents, words and classification classes
words = []
classes = []
documents = []
ignore_words = ['?']

# loop through each sentence in our intents patterns
for intent in intents['intents']:
    for pattern in intent['patterns']:
        # tokenize each word in the sentence
        w = nltk.word_tokenize(pattern)
        
        # add to our words list
        words.extend(w)
        
        # add to documents in our corpus
        documents.append((w, intent['tag']))
        
        # add to our classes list
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

# in linguistic morphology and information retrieval,
# stemming is the process of reducing inflected
# (or sometimes derived) words to their word stem, base or root form

# stem and lower each word and remove duplicates
words = [stemmer.stem(w.lower()) for w in words if w not in ignore_words]
words = sorted(list(set(words)))

# remove duplicate classes
classes = sorted(list(set(classes)))

print (len(documents), "documents", documents)
print (len(classes), "classes", classes)
print (len(words), "unique stemmed words", words)

# unfortunately this data structure won’t work
# with Tensorflow, we need to transform it further:
# from documents of words into tensors of numbers.

# create our training data
training = []
output = []

# create an empty array for our output
output_empty = [0] * len(classes)

# training set, bag of words for each sentence
for doc in documents:
    # initialize our bag of words
    bag = []

    # list of tokenized words for the pattern
    pattern_words = doc[0]

    # stem each word
    pattern_words = [stemmer.stem(word.lower()) for word in pattern_words]
    
    # create our bag of words array
    for w in words:
        if w in pattern_words:
            bag.append(1)
        else:
            bag.append(0)

    # output is a '0' for each tag and '1' for current tag
    output_row = list(output_empty)
    output_row[classes.index(doc[1])] = 1

    training.append([bag, output_row])

# shuffle our features and turn into np.array
random.shuffle(training)
training = np.array(training)

# create train and test lists
train_x = list(training[:,0])
train_y = list(training[:,1])

print(len(train_x), "train_x", train_x)
print(len(train_y), "train_y", train_y)

# train_x example: [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1] 
# this is one pattern (e.g. 'How are you') matched to the list of
# stemmed pattern words. For 'How are you', we expect three 1
# and everything else being 0.

# train_y example: [0, 0, 1, 0, 0, 0, 0, 0, 0]
# the output is the corresponding tag for that pattern
# (e.g. 'How are you' is tagged 'greeting') and we expect
# one 1 only with everything else being 0.

# we’re ready to build our Tensorflow model.

# build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# start training (apply gradient descent algorithm)
model.fit(train_x, train_y, n_epoch=1000, batch_size=8, show_metric=True)
model.save('model.tflearn')

# save all of our data structures
import pickle
pickle.dump( {'words':words, 'classes':classes, 'train_x':train_x, 'train_y':train_y}, open('training_data', 'wb'))
