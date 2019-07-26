#Sees if the triples created are the same as the good_image

import math
import mygrad as mg
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity


def accuracy(goodSim, badSim):
    """
    description: provides the accuracy of which good images are better. 
    :param goodSim: [np.array] of  length of batch size. Connected with the embedded text
    :param badSim:  [np.array] of length of batch size. Connected with the embedded text
    :return: the average of the correct responses
    """

    if isinstance(goodSim, mg.Tensor):
        goodSim = goodSim.data
    if isinstance(badSim, mg.Tensor):
        badSim = badSim.data
    return np.mean(goodSim>badSim)



def normalize(arr):
    """
    description:
        It should take in an array and normalize it  by dividing by the magnitude of the vector.
        The resulting array is the unit vector
    :param arr: [np.ndarray] shape = (M, 50)
    :return: [np.array] shape = (M, 50)
    """
    return arr/(mg.sqrt(mg.sum(arr**2)))

def similarity(caption, image):
    return mg.sum((caption*image), axis = 1)