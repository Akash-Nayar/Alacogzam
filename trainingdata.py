import numpy as np
import random
import secrets

def generate_training(captions, image_embeds, n):
	"""
	Generates triples to train the RNN

	Parameters:
	-----------
	captions : dict
		image_id : List[captions for that image]

	imaged_embeds : dict
		image_id : np.array(Embed for that image)
		
	n : int
		The number of triples to create.

	Returns:
		list : n number of triples.
	"""

	triples = []

	for i in range(n):
		good_key = random.choice(dictionary.keys())

		good_image = image_embeds[good_image]
		good_caption = secrets.choice(captions[good_key])
		index = random.choice(captions.keys())
		while index != good_key:
			index = random.choice(captions.keys())
		bad_image = image_embeds[index]
		triples.append(good_caption, good_image, bad_image)

	return triples



