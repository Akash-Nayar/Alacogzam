import glove
import numpy as np

import pickle
with open("captions.pickle", "rb") as database:
    captions = pickle.load(database)

captions = dict(captions)
import random

def find_caption():
    key = random.choice(list(captions))
    caption = captions[key][np.random.randint(5)]
    return key, caption

