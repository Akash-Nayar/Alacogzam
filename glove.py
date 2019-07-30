from collections import defaultdict
import numpy as np
import pickle
import gensim

from gensim.models.keyedvectors import KeyedVectors

path = r".\glove.6B.50d.txt.w2v"
glove = KeyedVectors.load_word2vec_format(path, binary=False)

with open("idfs.pickle", mode="rb") as opened_file:
    word_idfs_file = pickle.load(opened_file)

def glover(text):
    """
    :param
    text:string of our captions
    :return
    vectors: a (50,)np.ndarray that represents out caption in semantic space
    """


    vectors = np.zeros((50,))
    text = text.split()
    for word in text:

        if word not in glove:
            pass
        else:
            word_vector = glove[word]

            word_idf = word_idfs_file[word]

            vector = word_vector*word_idf

            vectors += vector
    return vectors.reshape((1,50))