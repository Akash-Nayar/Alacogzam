import glove
import numpy as np
import pickle
import json

with open("resnet18_features.pkl", mode="rb") as imgdata:
    images = pickle.load(imgdata)

images = dict(images)

import pickle
with open("captions.pickle", "rb") as database:
    captions = pickle.load(database)

captions = dict(captions)
import random

def find_caption():
    key = random.choice(list(captions))
    caption = captions[key][np.random.randint(5)]
    return key, caption

key, caption = find_caption()
embeddng_train_text = glove.glover(caption)
#key is the image id to the good image that we will use in our training process