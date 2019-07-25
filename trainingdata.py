import numpy as np
import random
import secrets

def generate_training(n):
	"""
	Generates triples to train the RNN

	Parameters:
	-----------
	n : int
		The number of triples to create.

	Returns:
		list : n number of triples.
	"""

	#REPLACE THIS
	triples = []

	for i in range(n):
		good_key = random.choice(dictionary.keys())

		good_image = image_embeds[good_image]
		good_caption = secrets.choice(dictionary[good_key])
		index = random.choice(dictionary.keys())
		while index != good_key:
			index = random.choice(dictionary.keys())
		bad_image = image_embeds[index]
		triples.append(good_caption, good_image, bad_image)
	return triples



