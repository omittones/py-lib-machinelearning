from timeit import default_timer
import itertools
from math import sqrt, ceil, sin, cos, pi
# from keras import initializers
# from keras.datasets import mnist
# from keras.optimizers import Adadelta
# from keras.utils import to_categorical
# from keras.models import Model
# from keras.layers import Input, Dense, LeakyReLU, Softmax, Dropout, Flatten, Conv2D, Reshape, MaxPooling2D, Activation, ReLU, ELU, PReLU
# from keras.losses import categorical_crossentropy
# from keras.metrics import categorical_accuracy
# from keras.callbacks import LearningRateScheduler
# from keras.preprocessing.image import ImageDataGenerator
# import keras.backend as K
# import tensorflow as tf
# import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
# from controllers import UserControlledLearningRate, Stopwatch
from PIL import Image, ImageDraw


def build_model(input_tensor=None):

    rn = initializers.glorot_normal()
    activation = ELU(alpha=0.8)

    inputs = Input(shape=(28, 28, 1), dtype='float32', tensor=input_tensor)
    x = inputs
    l = Conv2D(filters=32,
                kernel_size=(4, 4),
                strides=(1,1),
                padding='same',
                use_bias=True,
                kernel_initializer=rn,
                bias_initializer=rn)
    x = l(x)
    l = MaxPooling2D(pool_size=(2, 2))
    x = l(x)
    l = activation
    x = l(x)
    l = Conv2D(filters=32,
               kernel_size=(4, 4),
               strides=(1,1),
               padding='same',
               use_bias=True,
               kernel_initializer=rn,
               bias_initializer=rn)
    x = l(x)
    l = MaxPooling2D(pool_size=(2, 2))
    x = l(x)
    l = activation
    x = l(x)
    l = Flatten()
    x = l(x)
    l = Dense(500, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = activation
    x = l(x)
    l = Dense(250, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = activation
    x = l(x)
    # l = Dropout(rate=0.01)
    # x = l(x)
    l = Dense(10, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = Softmax()
    outputs = l(x)
    return Model(inputs=inputs, outputs=outputs)


def generate_images(nm_samples=100, nm_strokes_per_sample = 10):
    x_data = list()
    y_data = list()
    for _ in range(0, nm_samples):
        points = list([[0,0]])
        headings = list()
        heading = random.rand() * 2 * pi
        angles = (random.rand(nm_strokes_per_sample) - 0.5) * pi
        sizes = random.randint(0, 10, nm_strokes_per_sample)
        for (a,s) in zip(angles, sizes):
            heading += a
            x = points[-1][0] + cos(heading) * s
            y = points[-1][1] + sin(heading) * s
            points.append([x,y])
            headings.append(heading)

        points = np.array(points, dtype='float32')
        points = points - points.min(axis=(0))
        points = points / points.max(axis=(0))
        points = list((points * 20 + 3).flatten())

        image = Image.new('L', (28,28), 0)
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=255, width=2, joint='curve')
        data = np.asarray(image.getdata(), dtype='float32')
        data /= 255
        x_data.append(data)
        y_data.append(headings)

    return (x_data, y_data)


def show_images(images):
    i = 0
    figure = plt.figure()
    for img in images:
        i += 1
        ax = figure.add_subplot(10, 10, i)
        ax.imshow(img.reshape((28,28)), cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        if i == 100: break
    plt.show()


def main(preview_data = True):
    x, y = generate_images()
    show_images(x)
    return

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    multiplier = np.random.uniform(0, 1, x_train.shape[0])
    noise = np.random.normal(0.2, 0.2, x_train.shape)
    x_train += multiplier[:, None, None, None] * noise

    if preview_data:
        show_images(x_train)


    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)
    K.set_session(sess)

    model = None
    with tf.device('/GPU:0'):
        model = learn(x_train, y_train, x_test, y_test)
        test(model, x_test, y_test)
        #optimize(x_train, y_train)

    if model and preview_data:
        show_failures(model, x_test, y_test)