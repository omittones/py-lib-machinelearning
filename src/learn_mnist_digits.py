from timeit import default_timer
import itertools
from math import sqrt, ceil
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import Adadelta
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Dropout, Flatten, Conv2D, Reshape, MaxPooling2D, Activation, ReLU, ELU, PReLU
from keras.losses import categorical_crossentropy
from keras.metrics import categorical_accuracy
from keras.callbacks import LearningRateScheduler
from keras.preprocessing.image import ImageDataGenerator
import keras.backend as K
import tensorflow as tf
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from controllers import UserControlledLearningRate, Stopwatch


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
    l = Conv2D(filters=8,
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
    l = Dense(600, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = activation
    x = l(x)
    l = Dense(300, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = activation
    x = l(x)
    l = Dropout(rate=0.5)
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
    show_images(transformed)


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


def unstack(sample):
    bx, by = sample
    bx = bx[:,:,:,1]
    return (bx, by)


def optimize(x, y):
    #x = x[0:1000]
    #y = y[0:1000]
    for hyper in [0.0, 0.0001, 0.0002, 0.0004, 0.0008]:
        with tf.device('/GPU:0'):
            model = build_model()
            optimizer = Adadelta(decay=0)
            model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
            start = default_timer()
            result = model.fit(
                x=x,
                y=y,
                epochs=50,
                verbose=1,
                batch_size=132,
                shuffle=True)
            duration = default_timer() - start
            gy = result.history['loss']
            gx = np.linspace(0, duration, num=len(gy), endpoint=True)
            plt.plot(gx, gy, label=f'version-{hyper}')
            print(hyper, gy[-1])
    plt.ylabel('Loss')
    plt.xlabel('Time (sec)')
    plt.legend()
    plt.show()


def learn(x_train, y_train, x_test, y_test):
    controller = UserControlledLearningRate()

    model = build_model()
    model.summary()
    optimizer = Adadelta(lr=0.4)
    model.compile(optimizer=optimizer, loss=categorical_crossentropy, metrics=['accuracy'])
    model.fit(
        x=x_train,
        y=y_train,
        validation_data=(x_test, y_test),
        epochs=1000,
        verbose=1,
        batch_size=132,
        callbacks=[controller],
        shuffle=True)

    stats = model.test_on_batch(x_test, y_test)
    stats = dict(zip(model.metrics_names, stats))
    print('\nTest loss:', stats['loss'])
    print('Test error rate:', (1 - stats['acc']) * 100, '%')
    return model


def main(preview_data = False):

    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    multiplier = np.random.uniform(0, 1, x_train.shape[0])
    noise = np.random.normal(0.2, 0.2, x_train.shape)
    x_train += multiplier[:, None, None, None] * noise

    if preview_data:
        show_images(x_train)

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

    model = None
    with tf.device('/GPU:0'):
        #model = learn(x_train, y_train, x_test, y_test)
        optimize(x_train, y_train)

    if model and preview_data:
        show_failures(model, x_test, y_test)