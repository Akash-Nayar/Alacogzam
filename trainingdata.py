import numpy as np
import random
import secrets
import glove

def generate_training(captions, image_features, n):
	"""
	Generates triples to train the RNN

	Parameters:
	-----------
	captions : dict
		image_id : List[captions for that image]

	image_embeds : dict
		image_id : np.array(Embed for that image)
		
	n : int
		The number of triples to create.

	Returns:
		list : n number of triples.
	"""

	good_captions = []
	good_images = []
	bad_images = []

	for i in range(n):
		good_key = random.choice(captions.keys())
		good_image = image_features[good_key]
		good_caption = glove.glover(secrets.choice(captions[good_key]))
		index = random.choice(captions.keys())
		while index != good_key:
			index = random.choice(captions.keys())
		bad_image = image_features[index]
		good_captions.append(good_caption)
		good_images.append(good_image)
		bad_images.append(bad_image)

	return (np.array(good_captions), np.array(good_images), np.array(bad_images))



