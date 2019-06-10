import tensorflow as tf
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import Adadelta, Optimizer
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Dropout, Flatten, Conv2D, Reshape, MaxPooling2D, Activation, ReLU, ELU, PReLU, BatchNormalization
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import LearningRateScheduler, TensorBoard
from keras.preprocessing.image import ImageDataGenerator
from timeit import default_timer
import itertools
from math import sqrt, ceil
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from os import path
from controllers import UserControlledLearningRate, Stopwatch, StopOnEscape
from utils import batchify, show_images, show_failures
import keras.backend as K


def build_model(input_tensor=None, init_to_zeros = False):

    def activation():
        return ELU(alpha=0.8)

    if init_to_zeros:
        ki = initializers.zero()
        bi = initializers.zero()
    else:
        ki = initializers.glorot_normal()
        bi = initializers.zero()

    inputs = Input(shape=(28, 28, 1), dtype='float32', tensor=input_tensor)
    x = inputs
    x = Conv2D(filters=32,
               kernel_size=(6, 6),
               strides=(2,2),
               padding='same',
               use_bias=True,
               kernel_initializer=ki,
               bias_initializer=bi)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = BatchNormalization()(x)
    x = activation()(x)
    x = Conv2D(filters=32,
               kernel_size=(4, 4),
               strides=(1,1),
               padding='same',
               use_bias=True,
               kernel_initializer=ki,
               bias_initializer=bi)(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    x = activation()(x)
    x = Flatten()(x)
    x = Dense(500, kernel_initializer=ki, bias_initializer=bi)(x)
    x = activation()(x)
    x = Dense(250, kernel_initializer=ki, bias_initializer=bi)(x)
    x = activation()(x)
    x = Dropout(rate=0.9)(x)
    x = Dense(10, kernel_initializer=ki, bias_initializer=bi)(x)
    x = Softmax()(x)
    outputs = x
    return Model(inputs=inputs, outputs=outputs)


def prepare_data(x, y):
    y = to_categorical(y, 10).astype('float32')
    x = x / 255
    x = x.reshape((-1, 28, 28, 1)).astype('float32')
    return (x, y)


def preview_transformations(generator, images):
    images = images[0:100]
    transformed = next(generator.flow(images, batch_size=100))
    show_images(transformed)


def optimize(x, y):
    # x = x[0:20000]
    # y = y[0:20000]
    try:
        for hyper in [500, 1000, 2000, 4000, 8000]:
            model = build_model()
            optimizer = Adadelta(decay=0)
            model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
            start = default_timer()
            result = model.fit(
                x=x,
                y=y,
                epochs=10,
                verbose=1,
                batch_size=hyper,
                callbacks=[StopOnEscape()],
                shuffle=True)
            duration = default_timer() - start
            gy = result.history['loss']
            gx = np.linspace(0, duration, num=len(gy), endpoint=True)
            plt.plot(gx, gy, label=f'version-{hyper}')
            print(hyper, gy[-1])
    except Exception as ex:
        print(ex)
    plt.ylabel('Loss')
    plt.xlabel('Time (sec)')
    plt.legend()
    plt.show()


def learn(x_train, y_train, x_val, y_val, clean=True):
    controller = UserControlledLearningRate()

    model = build_model()
    model.summary()

    # statsdir = path.join(__file__, '../../logs/mnist')
    # tb = TensorBoard(log_dir=statsdir, histogram_freq=1)

    filename = path.join(__file__, '../learn_mnist_digits.h5')
    if not clean and path.exists(filename):
        print(f'Loading weights from {filename}')
        model.load_weights(filename)

    optimizer = Adadelta(decay=0)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_val, y_val),
        epochs=1000,
        verbose=1,
        batch_size=8000,
        callbacks=[controller],
        shuffle=True)

    print(f'\nSaving weights to {filename}')
    model.save_weights(filename)

    return model


def test(model, x, y):
    loss = 0
    acc = 0
    for b in batchify(x, y):
        stats = model.test_on_batch(b[0], b[1])
        stats = dict(zip(model.metrics_names, stats))
        loss += stats['loss'] * b[0].shape[0]
        acc += (1 - stats['acc']) * 100 * b[0].shape[0]
    loss = loss / x.shape[0]
    acc = acc / x.shape[0]
    print('\nTest loss:', loss)
    print('Test error rate:', acc, '%')


def main(preview_data = False):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    x_train = np.concatenate((x_train, x_train))
    y_train = np.concatenate((y_train, y_train))

    multiplier = np.random.uniform(0, 1, x_train.shape[0])
    noise = np.random.normal(0.2, 0.2, x_train.shape)
    x_train += multiplier[:, None, None, None] * noise

    # with Stopwatch('Preparing data'):
    #     generator = ImageDataGenerator(
    #          rotation_range=30,
    #          zoom_range=0,
    #          width_shift_range=4,
    #          height_shift_range=4,
    #          shear_range=10,
    #          brightness_range=None,
    #          fill_mode='nearest',
    #          data_format='channels_last')
    #     data = generator.flow(x_train, y_train, batch_size=len(x_train))
    #     x_gen, y_gen = next(data)
    #     x_train = np.concatenate((x_train, x_gen), axis=0)
    #     y_train = np.concatenate((y_train, y_gen), axis=0)

    if preview_data:
        show_images(x_train)

    model = None
    with tf.device('/device:GPU:0'):

        config = tf.ConfigProto(log_device_placement=False)
        sess = tf.Session(config=config)
        K.set_session(sess)

        model = learn(x_train, y_train, x_test, y_test)
        test(model, x_test, y_test)
        # optimize(x_train, y_train)

        if model and preview_data:
            show_failures(model, x_test, y_test)