#!/usr/bin/env python3
"""
    :author: James Jenkins
"""

import numpy as np
import pickle
import collections
import pprint

# Load little mnist data
train, validate, test = np.array(pickle.load(open("littlemnist.pkl", "rb"), encoding = "latin1"))

# Add columns of 1s to the train, validate, and test sets
train[0] = np.c_[train[0], np.ones((len(train[0]), 1))]
validate[0] = np.c_[validate[0], np.ones((len(validate[0]), 1))]
test[0] = np.c_[test[0], np.ones((len(test[0]), 1))]

# Set learning rate (TODO: Maybe start this higher and decay in relation to err)
alpha = 0.4

n_classes = 10
dim_data  = len(train[0][0])

# Initialize random weights (+1 to dim_data for bias)
w = np.random.rand(n_classes, dim_data)

# A zero vector for comparison later
z10 = np.zeros((n_classes,1))


def Classify(example):
    return w.dot(example)


def OneHot(index, dim = 10):
    """
    Converts an index into a one-hot encoded column vector.
    """
    a = np.zeros((dim,1))
    a[index] = 1
    return a


def Validate():
    """
    Runs through all the validation data to get an error estimate.
    """
    correct = sum((np.argmax(Classify(x))) == validate[1][x_i] for x_i, x in enumerate(validate[0]))
    print("Validation set correct: %d"%correct)


Validate()

try:
    iteration = 0
    while True:
        correct = 0
        # Enumerate training examples
        for x_i, x in enumerate(train[0]):
            y = Classify(x)
            d = OneHot(train[1][x_i])

            # Move a proportion (alpha) of the difference between where we want to be
            # and where we are
            delta = d - OneHot(np.argmax(np.reshape(y, (10,1))))
            #delta = d - np.reshape(y, (10,1))
            w += alpha * delta * x
            correct += np.alltrue(delta == z10)

        if iteration % 10 == 0:
            print("Train right: %d"%correct)
            Validate()

        # Break out once we achieve max on train data
        if correct == 10000:
            break

        iteration += 1

except KeyboardInterrupt:
    print()

# Calculate results on the test set
confusion = collections.defaultdict(lambda: [0]*10)
errors = 0
for x_i, x in enumerate(test[0]):
    y = np.argmax(Classify(x))
    confusion[test[1][x_i]][y] += 1
    errors += test[1][x_i] != y

pprint.pprint(confusion)

print("Test set error: %f"%(errors/len(test[0])))


#with open("weights.bin", "wb") as f:
#    pickle.dump(w, f)
