import matplotlib.pyplot as plt
from math import sqrt, ceil, sin, cos, pi, floor
import numpy as np


def show_failures(model, x, y, shape=(28,28)):
    fails = []
    for batch in batchify(x, y):
        prediction = model.predict_on_batch(batch[0])
        pi = np.argmax(prediction, axis=1)
        yi = np.argmax(batch[1], axis=1)
        fails.extend([x for x, y, t in zip(batch[0], yi, pi) if y != t])
    show_images(fails, shape=shape)


def show_images(images, labels = None, shape=(28,28)):
    i = 0
    x = floor(sqrt(min(len(images), 100)))
    y = x
    if x * y < len(images):
        x += 1
    if x * y < len(images):
        y += 1
    figure = plt.figure()
    for img in images:
        label = None if labels == None else labels[i]
        i += 1
        ax = figure.add_subplot(x, y, i, label=label)
        ax.imshow(img.reshape(shape), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == x * y: break
    plt.show()


def batchify(x, y, size = 500):
    """Split data intp batches and return iterable.

    Parameters
    ----------
    x : indexable input data

    y : indexable expected output data

    size : integer, optional, default: 500
        size of batches

    Returns
    -------
    (x, y) : yield tuples of batches
    """

    total = x.shape[0] - 1
    start = 0
    while start < total:
        end = min(start + size, total)
        yield (x[start:end], y[start:end])
        start = end + 1