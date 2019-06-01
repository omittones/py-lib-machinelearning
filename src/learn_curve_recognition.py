import itertools
from timeit import default_timer
from math import sqrt, ceil, sin, cos, pi, floor
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
from utils import show_images, batchify

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
        angles = list()
        heading = random.rand() * 2 * pi
        a = 0
        for _ in range(0, nm_strokes_per_sample):
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
        points = list((points * 21 + 3).flatten())

        image = Image.new('L', (28,28), 0)
        draw = ImageDraw.Draw(image)
        draw.line(points, fill=255, width=2)
        data = np.asarray(image.getdata(), dtype='float32')
        data /= 255
        x_data.append(data)
        y_data.append(angles)

    return (x_data, y_data)


def main(preview_data = True):
    x, y = generate_images(nm_samples=100)
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