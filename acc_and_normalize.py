#Sees if the triples created are the same as the good_image

import math
import mygrad as mg
from sklearn.metrics.pairwise import cosine_similarity


# compute cosine-similarity between all pairs of rows of `x`

#The triples is a tuple of size 3
#In each iteration, it has a numpy array of size n
# 1: embedded text
# 2: embedded good_image
# 3: embedded bad_image

#Ask Lillian after I set it based on the generating triples code
def accuracy_triples(triples, n):
    """
    :param emb_text: the semantic embedding of text. [List of numpy arrays]
    :param emb_good: good_image embedding; numpy array [List of numpy arrays]
    :param emb_bad: bad_image embedding; numpy array   [List of numpy arrays]
    :param n: the numbers of triples that have to be processed
    :return:
    """

    emb_text = triples[0]
    emb_good = triples[1]
    emb_bad = 0

    num_correct = 0
    for i in range(n):

        sim_to_good = cosine_similarity(emb_text, emb_good)
        sim_to_bad = cosine_similarity(emb_text, emb_bad)
        sim_or_not = sim_to_good > sim_to_bad
        if sim_or_not:
            num_correct += 1
        else:
            num_correct += 0

        percentage = num_correct / n
        percentage = math.round(percentage, 2)

    return percentage


def normalize(arr):
    """
    description:
        It should take in an array and normalize it  by dividing by the magnitude of the vector.
        The resulting array is the unit vector
    :param arr: [np.ndarray] shape = (M, 50)
    :return: [np.array] shape = (50,) Think
    """
    return arr/mg.sqrt((arr**2).sum())