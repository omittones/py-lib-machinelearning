import time
import itertools
from math import sqrt, ceil
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Dropout, Flatten, Conv2D, Reshape, MaxPooling2D, Activation
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from controllers import UserControlledLearningRate, Stopwatch


def build_model(filters=8, kernel_size=4, pool_size=2):
    rn = initializers.glorot_normal()
    inputs = Input(shape=(28, 28, 1), dtype='float32')
    x = inputs
    l = Conv2D(filters=filters,
                kernel_size=(kernel_size, kernel_size),
                strides=(1,1),
                padding='same',
                use_bias=True,
                kernel_initializer=rn,
                bias_initializer=rn)
    x = l(x)
    l = MaxPooling2D(pool_size=(pool_size, pool_size))
    x = l(x)
    l = Flatten()
    x = l(x)
    l = Dense(100, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = LeakyReLU(alpha=0.3)
    x = l(x)
    l = Dense(50, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = LeakyReLU(alpha=0.3)
    x = l(x)
    l = Dropout(rate=0.3)
    x = l(x)
    l = Dense(10, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = Softmax()
    outputs = l(x)
    return Model(inputs=inputs, outputs=outputs)


def prepare_data(x, y):
    y = to_categorical(y, 10).astype('float32')
    x = x / 255
    x = x.reshape((-1, 28, 28, 1)).astype('float32')
    return (x, y)


def preview_transformations(generator, images):
    images = images[0:100]
    transformed = next(generator.flow(images, batch_size=100))
    figure = plt.figure()
    i = 0
    for img in transformed:
        i += 1
        ax = figure.add_subplot(10, 10, i)
        ax.imshow(img, cmap='gray')
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def unstack(sample):
    bx, by = sample
    bx = bx[:,:,:,1]
    return (bx, by)


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


def show_failures(model, x, y):
    prediction = model.predict_on_batch(x)
    pi = np.argmax(prediction, axis=1)
    yi = np.argmax(y, axis=1)
    fails = [x for x, y, t in zip(x, yi, pi) if y != t]
    show_images(fails)


def optimize(x, y):
    x = x[0:60000]
    y = y[0:60000]
    for hyper in [0.008, 0.016, 0.032, 0.064]:
        with tf.device('/GPU:0'):
            model = build_model(filters=8, kernel_size=4, pool_size=2)
            optimizer = Adadelta(lr=0.08)
            model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
            result = model.fit(
                x=x,
                y=y,
                epochs=5,
                verbose=1,
                batch_size=128,
                shuffle=True)

            plt.plot(result.history['loss'], label=f'{hyper}-loss')
            print(hyper, result.history['loss'][-1])

    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()
    plt.show()


def learn(x_train, y_train, x_test, y_test):
    controller = UserControlledLearningRate()
    epochs = 500
    batch_size = 100
    with tf.device('/GPU:0'):
        model = build_model(filters=8, kernel_size=4, pool_size=2)
        optimizer = Adadelta(lr=0.08)
        model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
        model.fit(
            x=x_train,
            y=y_train,
            validation_split=0.15,
            epochs=epochs,
            verbose=1,
            batch_size=batch_size,
            callbacks=[controller],
            shuffle=True)

        stats = model.test_on_batch(x_test, y_test)
        stats = dict(zip(model.metrics_names, stats))
        print('Test loss:', stats['loss'])
        print('Test error rate:', (1 - stats['acc']) * 100, '%')
        # show_failures(model, x_test, y_test)


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    multiplier = np.random.uniform(0, 1, x_train.shape[0])
    noise = np.random.normal(0.2, 0.2, x_train.shape)
    x_train += multiplier[:, None, None, None] * noise
    show_images(x_train)

    x = np.concatenate([x_train, x_test])
    y = np.concatenate([y_train, y_test])

    # with Stopwatch('Preparing data'):
    #     generator = ImageDataGenerator(
    #          rotation_range=30,
    #          zoom_range=0,
    #          width_shift_range=4,
    #          height_shift_range=4,
    #          shear_range=10,
    #          #brightness_range=(0, 1),
    #          fill_mode='nearest',
    #          data_format='channels_last')
    #     x = np.stack([x, x, x], axis = 3)
    #     preview_transformations(generator, x)
    #     data = generator.flow(x, y, batch_size=size)
    #     x, y = next(map(unstack, data))

    config = tf.ConfigProto(log_device_placement=False)
    sess = tf.Session(config=config)
    K.set_session(sess)

    learn(x, y, x_test, y_test)



    # model.fit_generator(
    #     data,
    #     steps_per_epoch=size/batch_size,
    #     epochs=epochs,
    #     verbose=1,
    #     callbacks=[controller])

    # train(100, 100, 10)
    # train(100, 1000, 100)
    # train(100, 10000, 1000)
    # train(100, 60000, 6000)
    # Elapsed 78.5720272064209 seconds
    # Test error rate: 4.250001907348633 %

    # train(80, 60000, 100)
    # Elapsed 77.50399446487427 seconds
    # Test error rate: 2.0299971103668213 %