#!/usr/bin/env python3

import numpy as np
import pickle

train, validate, test = np.array(pickle.load(open("littlemnist.pkl", "rb"), encoding = "latin1"))

# Set out learning rate
alpha = 0.5

n_classes = 10
dim_data  = len(train[0][0])

# Initialize random weights (+1 to dim_data for bias)
w = np.random.rand(n_classes, dim_data + 1)
#w = np.zeros((n_classes, dim_data + 1))

train[0] = np.c_[train[0], np.ones((len(train[0]), 1))]
validate[0] = np.c_[validate[0], np.ones((len(validate[0]), 1))]


def classify(example):
    return w.dot(example)

def oneHot(index):
    a = np.zeros((10,1))
    a[index] = 1
    return a

def Validate():
    correct = sum((np.argmax(classify(x))) == validate[1][x_i] for x_i, x in enumerate(validate[0]))
    print("Correct: %d"%correct)

#error = np.sum(x[:,1] - list(map(classify, x[:,0])))/len(x)
Validate()
z10 = np.zeros((10,1))

iteration = 0
while True:
    correct = 0
    # Enumerate training examples
    for x_i, x in enumerate(train[0]):
        y = classify(x)
        d = oneHot(train[1][x_i])

        # Move a proportion (alpha) of the difference between where we want to be
        # and where we are
        delta = d - oneHot(np.argmax(np.reshape(y, (10,1))))
        #delta = d - np.reshape(y, (10,1))
        w += alpha * delta * x
        correct += np.alltrue(delta == z10)
        #error = np.sum(x[:train_samples,1] - list(map(classify, x[:train_samples,0])))/len(x)

    if iteration % 10 == 0:
        print("Train right: %d"%correct)
        Validate()

    iteration += 1


#right = 0
#for ex in x[train_samples:]:
#    right += classify(ex[0]) == ex[1]
#
#print("len(test_x) = {}".format(len(x) - train_samples))
#print("right = {}".format(right))
