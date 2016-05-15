#!/usr/bin/env python3
import numpy as np
import pickle
from PIL import Image

w = pickle.load(open("weights1000.pkl", "rb"))

def Classify(example):
    return w.dot(example)

#Seems to get 2, 3, 4 correct...
for i in range(0, 5):
    image = Image.open("test_images/{}.jpg".format(i)).convert("L")
    x = np.asarray(image.getdata())
    x = (255 - x)/255
    x = np.r_[x, 1]

    y = Classify(x)
    print(y)
    print("Actual: {} Classification: {}".format(i, np.argmax(y)))
