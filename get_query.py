import glove
import numpy as np

def query():
    text = input("Search: ")
    query_vec = glove.glover(text)

    return query_vec