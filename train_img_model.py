from img_model import Model
import mygrad as mg
import mynn
import numpy as np
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
from mynn.optimizers.adam import Adam
import trainingdata as td
import split_n_stitch as ss
import training_functions as tf

#Call the model to convert the image features to image embeddings.
se_img = Model()
optim = Adam(model.parameters, learning_rate=0.1)

#Unpickle the captions database
with open("captions.pickle", mode="rb") as captiondata:
    captions = dict(pickle.load(captiondata))

#Unpickle the image database
with open("resnet18_features.pkl", mode="rb") as imgdata:
    img = dict(pickle.load(imgdata))

#The captions and image embeddings are used to generate 16000 triples.
#Each triple is in the format of (caption, good image embedding, bad image embedding)

captions, good_images, bad_images = td.generate_training(captions, img, 64000)
#testingtriples = td.generate_training(captions, image_embeds, 16000)

#The number of epochs and the batch size are defined
num_epochs = 500
batch_size = 25

for epoch_cnt in range(num_epochs):
    idxs = np.arange(len(captions))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(captions) // batch_size):
        batch_indices = idxs[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
        caption_batch = tf.normalize(captions[batch_indices])
        good_batch = tf.normalize(se_img(good_images[batch_indices]))
        bad_batch = tf.normalize(se_img(bad_images[batch_indices]))

        goodSim = tf.similarity(caption_batch, good_batch)
        badSim = tf.similarity(caption_batch, bad_batch)

        acc = tf.accuracy(goodSim, badSim)

        loss = margin_ranking_loss(goodSim, badSim, 1, margin=0.1)
        loss.backward()

        optim.step()
        loss.null_gradients()

        plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)
