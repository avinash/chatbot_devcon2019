# Contextual Chatbots with Tensorflow
# https://chatbotsmagazine.com/contextual-chat-bots-with-tensorflow-4391749d0077

# we’ll build a simple state-machine to handle responses,
# using our intents model (from the previous step) as our classifier.
# that’s how chatbots work.

# things we need for NLP
import nltk
from nltk.stem.lancaster import LancasterStemmer
stemmer = LancasterStemmer()

# things we need for Tensorflow
import numpy as np
import tflearn
import tensorflow as tf
import random

# restore all of our data structures
import pickle
data = pickle.load(open('training_data', 'rb'))

words = data['words']
classes = data['classes']
train_x = data['train_x']
train_y = data['train_y']

# import our chat-bot intents file
import json
with open('intents.json') as json_data:
    intents = json.load(json_data)

# build neural network
net = tflearn.input_data(shape=[None, len(train_x[0])])
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, 8)
net = tflearn.fully_connected(net, len(train_y[0]), activation='softmax')
net = tflearn.regression(net)

# Define model and setup tensorboard
model = tflearn.DNN(net, tensorboard_dir='tflearn_logs')

# load our saved model
model.load('model.tflearn')

# before we can begin processing intents,
# we need a way to produce a bag-of-words from user input.
# this is the same technique as we used earlier to create
# our training documents.

def clean_up_sentence(sentence):
    # tokenize the pattern
    sentence_words = nltk.word_tokenize(sentence)
    
    # stem each word
    sentence_words = [stemmer.stem(word.lower()) for word in sentence_words]
    
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bag_of_words(sentence, words, show_details=False):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    
    # bag of words
    bag = [0] * len(words)  
    
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s: 
                bag[i] = 1
            if show_details:
                print ("found in bag: %s" % w)

    return(np.array(bag))

# print(len(words), "words", words)
# p = bag_of_words("is your shop open today?", words)
# print(len(p), "p", p)

# we are now ready to build our response processor.
# with one twist: it provides basic contextualization.

# create a data structure to hold user context
context = {}

ERROR_THRESHOLD = 0.25

def classify(sentence):
    # generate probabilities from the model
    results = model.predict([bag_of_words(sentence, words)])[0]
    
    # filter out predictions below a threshold
    results = [[i, r] for i, r in enumerate(results) if r > ERROR_THRESHOLD]
    
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    
    return_list = []
    for r in results:
        return_list.append((classes[r[0]], r[1]))

    # return tuple of intent and probability
    return return_list

def response(sentence, user_id='123', show_details=False):
    results = classify(sentence)
    
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            for i in intents['intents']:
                # find a tag matching the first result
                if i['tag'] == results[0][0]:
                    # set context for this intent if necessary
                    if 'context_set' in i:
                        if show_details:
                            print ('context:', i['context_set'])

                        context[user_id] = i['context_set']

                    # check if this intent is contextual and applies to this user's conversation
                    if not 'context_filter' in i or \
                    (user_id in context and 'context_filter' in i and i['context_filter'] == context[user_id]):
                        if show_details:
                            print ('tag:', i['tag'])
                        # a random response from the intent
                        return print(random.choice(i['responses']))

            results.pop(0)

# print('is your shop open today?')
# print(classify('is your shop open today?'))
# response('is your shop open today?')
# print()

# print('do you take cash?')
# print(classify('do you take cash?'))
# response('do you take cash?')
# print()

# print('what kind of mopeds do you rent?')
# print(classify('what kind of mopeds do you rent?'))
# response('what kind of mopeds do you rent?')
# print()

# print('we want to rent a moped')
# print(classify('we want to rent a moped'))
# response('we want to rent a moped')
# print()

# print('today')
# print(classify('today'))
# response('today')
# print()

# print('Goodbye, see you later')
# print(classify('Goodbye, see you later'))
# response('Goodbye, see you later')
# print()

for _ in range(50):
    print()

while True:
    print("=> ", end="")
    question = input()

    if question == "0":
        break

    response(question)
    print()
