import time
import itertools
from math import sqrt, ceil
from keras import initializers
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Reshape
from keras.losses import categorical_crossentropy
from keras.preprocessing.image import ImageDataGenerator
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from controllers import UserControlledLearningRate

def build_model():
    rn = initializers.glorot_uniform()
    inputs = Input(shape=(28, 28), dtype='float32')
    x = inputs
    l = Reshape((28*28,))
    x = l(x)
    l = Dense(100, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = LeakyReLU(alpha=0.3)
    x = l(x)
    l = Dense(50, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = LeakyReLU(alpha=0.3)
    x = l(x)
    l = Dense(10, kernel_initializer=rn, bias_initializer=rn)
    x = l(x)
    l = Softmax()
    outputs = l(x)
    return Model(inputs=inputs, outputs=outputs)


def prepare_data(x, y):
    x = x / 255
    y = to_categorical(y, 10)
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


def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    generator = ImageDataGenerator(
        rotation_range=60,
        zoom_range=0,
        width_shift_range=4,
        height_shift_range=4,
        shear_range=10,
        #brightness_range=(0, 1),
        fill_mode='nearest',
        data_format='channels_last')

    x_train = np.stack([x_train, x_train, x_train], axis = 3)
    preview_transformations(generator, x_train)

    model = build_model()
    controller = UserControlledLearningRate()

    def train(epochs, size, batch_size):

        sgd = SGD(lr=0.5, momentum=0.01, decay=0.0, nesterov=False)
        model.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])

        def unstack(sample):
            bx, by = sample
            bx = bx[:,:,:,1]
            return (bx, by)

        print('Preparing data...')
        data = generator.flow(x_train, y_train, batch_size=size)
        x, y = next(map(unstack, data))

        # model.fit_generator(
        #     data,
        #     steps_per_epoch=size/batch_size,
        #     epochs=epochs,
        #     verbose=1,
        #     callbacks=[controller])

        model.fit(x=x, y=y, epochs=epochs, verbose=2, batch_size=batch_size, callbacks=[controller], shuffle=True)


    start = time.time()

    train(500, 60000, 100)
    print('Elapsed', time.time() - start, 'seconds')

    final = model.test_on_batch(x_test, y_test)
    final = dict(zip(model.metrics_names, final))
    print('Test error rate:', (1 - final['acc']) * 100, '%')


    # train(100, 100, 10)
    # train(100, 1000, 100)
    # train(100, 10000, 1000)
    # train(100, 60000, 6000)
    # Elapsed 78.5720272064209 seconds
    # Test error rate: 4.250001907348633 %

    # train(80, 60000, 100)
    # Elapsed 77.50399446487427 seconds
    # Test error rate: 2.0299971103668213 %