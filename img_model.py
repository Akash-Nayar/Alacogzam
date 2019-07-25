import pickle
import json

import mygrad as mg
import mynn
import numpy as np
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
from mynn.optimizers.adam import Adam


class Model:
    def __init__(self):
        """ This initializes all of the layers in our model, and sets them
        as attributes of the model.

        """
        self.dense1 = dense(512, 50)

    def __call__(self, x):
        '''Passes data as input to our model, performing a "forward-pass".

        This allows us to conveniently initialize a model `m` and then send data through it
        to be classified by calling `m(x)`.

        Parameters
        ----------
        x : Union[numpy.ndarray, mygrad.Tensor], shape=(M, 50)
            A batch of data consisting of M pieces of data,
            each with a dimensionality of D_full.

        Returns
        -------
        mygrad.Tensor, shape=(M, 50)
            The model's prediction for each of the M pieces of data.
        '''
        return self.dense1(x)

    @property
    def parameters(self):
        """ A convenience function for getting all the parameters of our model.

        This can be accessed as an attribute, via `model.parameters`

        Returns
        -------
        Tuple[Tensor, ...]
            A tuple containing all of the learnable parameters for our model """
        return self.dense1.parameters

