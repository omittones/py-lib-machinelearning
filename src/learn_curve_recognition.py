import itertools
from timeit import default_timer
from math import sqrt, ceil, sin, cos, pi, floor
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Dropout, Flatten, Conv2D, Reshape, MaxPooling2D, Activation, ReLU, ELU, PReLU
from keras.losses import mean_squared_error
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from numpy import random
from controllers import UserControlledLearningRate, Stopwatch
from PIL import Image, ImageDraw
from utils import show_images, show_failures


def generate_images(nm_samples=100, nm_strokes_per_sample = 10, side_pixels = 100):
    x_data = list()
    y_data = list()
    for _ in range(0, nm_samples):
        points = list([[0,0]])
        angles = list()
        heading = random.rand() * 2 * pi
        a = 0
        total_strokes = random.randint(1, nm_strokes_per_sample)
        for _ in range(0, nm_strokes_per_sample):
            if _ >= total_strokes:
                angles.append(-100)
            else:
                if a == 0 or random.rand() > 0.8:
                    a = round((random.rand() - 0.5) * pi, 1)
                if random.rand() > 0.8:
                    a = -a
                heading += a
                x = points[-1][0] + cos(heading)
                y = points[-1][1] + sin(heading)
                points.append([x,y])
                angles.append(a)

        points = np.array(points, dtype='float32')
        points = points - points.min()
        points = points / points.max()
        points = list((points * (side_pixels - 6) + 3).flatten())

        image = Image.new('L', (side_pixels, side_pixels), 0)
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=255, width=2)
        data = np.asarray(image.getdata(), dtype='float32').reshape((side_pixels, side_pixels, 1))
        data /= 255
        x_data.append(data)
        y_data.append(angles)

    x_data = np.stack(x_data)
    y_data = np.stack(y_data)

    return (x_data, y_data)


def learn(x_train, y_train, x_test, y_test):
    controller = UserControlledLearningRate()

    model = build_model()
    model.summary()
    optimizer = Adadelta(decay=0)
    model.compile(optimizer=optimizer, loss=mean_squared_error)
    model.fit(
        x=x_train,
        y=y_train,
        epochs=1000,
        verbose=1,
        batch_size=160,
        callbacks=[controller],
        validation_data=(x_test, y_test),
        shuffle=True)
    return model


def build_model():

    rn = initializers.glorot_normal()
    def activation():
        return ELU(alpha=0.8)

    inputs = Input(shape=(28, 28, 1), dtype='float32')
    x = inputs
    l = Flatten()
    x = l(x)
    l = Dense(500, kernel_initializer=rn)
    x = l(x)
    l = activation()
    x = l(x)
    l = Dropout(rate=0.2)
    x = l(x)
    l = Dense(250, kernel_initializer=rn)
    x = l(x)
    l = activation()
    x = l(x)
    l = Dense(10, kernel_initializer=rn)
    outputs = l(x)
    return Model(inputs=inputs, outputs=outputs)


def main(preview_data = True):

    x, y = generate_images(nm_samples=70000, side_pixels=28)
    if preview_data:
        show_images(x[:100], [str(i) for i in y[:100]], shape=(28, 28))

    x_train = x[0:-10000]
    y_train = y[0:-10000]
    x_test = x[-10000:]
    y_test = y[-10000:]

    model = None
    with tf.device('/GPU:0'):

        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(config=config)
        K.set_session(sess)

        model = learn(x_train, y_train, x_test, y_test)
