from img_model import Model
import mygrad as mg
import mynn
import numpy as np
from mynn.layers.dense import dense
from mygrad.nnet import margin_ranking_loss
from mynn.optimizers.adam import Adam
model = Model()

optim = Adam(model.parameters, learning_rate=0.1)

num_epochs = 500
batch_size = 25

for epoch_cnt in range(num_epochs):
    idxs = np.arange(len(iris))
    np.random.shuffle(idxs)

    for batch_cnt in range(0, len(iris) // batch_size):
        batch_indices = idxs[batch_cnt * batch_size: (batch_cnt + 1) * batch_size]
        batch = iris[batch_indices]

        prediction = model(batch)
        truth = iris[batch_indices]

        loss = mean_squared_loss(prediction, truth)
        loss.backward()

        optim.step()
        loss.null_gradients()

        plotter.set_train_batch({"loss": loss.item()}, batch_size=batch_size)

    # epoch loss
    if epoch_cnt % 100 == 0:
        prediction = model(iris)
        truth = iris
        loss = mean_squared_loss(prediction, truth)
        print(f'epoch {epoch_cnt:5}, loss = {loss.item():0.3f}')
# </COGINST>