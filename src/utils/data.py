import matplotlib.pyplot as plt
from math import sqrt, ceil, sin, cos, pi, floor


def show_images(images):
    i = 0
    x = floor(sqrt(min(len(images), 100)))
    y = x
    if x * y < len(images):
        x += 1
    if x * y < len(images):
        y += 1
    figure = plt.figure()
    for img in images:
        i += 1
        ax = figure.add_subplot(x, y, i)
        ax.imshow(img.reshape((28,28)), cmap='gray')
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