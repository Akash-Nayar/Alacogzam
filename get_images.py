#Standard Python Packages
import numpy as np
import pickle
import json
from PIL import Image
import requests
from io import BytesIO

# Our own files
import get_query
import glove
import training_functions as fu
import split_n_stitch as shit


def get_images(k):
    """
    Display k most similar images to a given caption

    Parameters:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    k: int
    The number of images to return

    Returns:
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    list: length-k list of IDs
    """

    keys = np.linspace(0, 99, 50)
    values = np.random.randn(50, 50)
    image_embeddings = dict(zip(keys, values))

    # with open("image_embeddings.pickle", "rb") as f:
    # image_embeddings = pickle.load(f)

    with open("url.pickle", "rb") as urls:
        url = pickle.load(urls)

    # GETTING THE SIMILAR IDS
    caption_embed = get_query.query()  # embedded text
    caption_embed = fu.normalize(caption_embed)
    ids, embeddings = (shit.split(image_embeddings))
    for embed in embeddings:
        embed.reshape((50, 1))

    embed_values = []
    for e in embeddings:
        embed_values.append((caption_embed @ e).data)
    embed_values = np.array(embed_values)

    max_values = np.argsort(embed_values, axis=0)[:k].flatten()
    similar_image_ids = []

    for m in max_values:
        similar_image_ids.append(ids[m])
    # END ID ACQUISITION

    image_links = []
    for img_id in similar_image_ids:
        image_links.append(url[img_id])

    for i in image_links:
        response = requests.get(i)
        img = Image.open(BytesIO(response.content))

    return similar_image_ids