import pickle
import json

import mygrad as mg
import mynn
import numpy as np
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
from mynn.optimizers.adam import Adam

#with open("resnet18_features.pkl", mode="rb") as imgdata:
#    img = str(pickle.load(imgdata))
#print("loaded")

def split(images):
    """
    Takes in a dictionary of image ID's and the image features and splits into list of
    keys and an array of features, the features being defined as an (M, 512) numpy arrays.
    M represents the number of features in the dictionary.
    :param images: dict[int:numpy.ndarray(), shape = (1,512)]
    A dictionary of image ID's and features.
    :return: list of length M, numpy.ndarrray() where shape=(M, 512)
    Returns a tuple containing a list of image ID's and an array of features.
    """
    keys = [key for key in images]
    features = np.array([images[key] for key in keys])
    return keys, features

def stitch(keys, embeddings):
    """
    Takes in a list of image ID's and an array of image embeddings and combines them into a dictionary.
    The image ID's will be the keys and the embeddings will be the values in the dictionary.
    :param keys: list(int)
    A list of integer image ID's
    :param embeddings: numpy.ndarray(), shape = (M, 50)
    An array of embeddings after the features have been put into the model and converted
    from size (1, 512) to size (1, 50).
    M represents the number of embeddings in the array.
    :return: dict[int: numpy.ndarray(), shape =(1, 50)]
    """
    dictionary = {keys[i]: embeddings[i] for i in range(len(keys))}
    return dictionary


