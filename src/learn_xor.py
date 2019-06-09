import time
import numpy as np
import seaborn as sns
from os import path
import io
from keras import activations, initializers
from keras.layers import Activation, Dense, Input, Softmax, ReLU
from keras.losses import binary_crossentropy
from keras.models import Model
from keras.optimizers import SGD, Adadelta
from matplotlib import pyplot
from controllers import UserControlledLearningRate
import tensorflow as tf

def draw_seaborn_scatter(data, prediction):
    sns.set(style="darkgrid")

    p = sns.blend_palette(['#ff0000','#ff0000','#0000ff','#0000ff'], as_cmap=True)
    _, ax = pyplot.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=prediction[:,0], palette=p)
    pyplot.show()


def draw_seaborn_density(data, prediction):

    reds = data[prediction[:,0]>0.5,:]
    blues = data[prediction[:,1]>0.5,:]

    sns.set(style="darkgrid")
    _, ax = pyplot.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Draw the two density plots
    ax = sns.kdeplot(reds[:,0], reds[:,1], cmap="Reds", shade=True, shade_lowest=False)
    ax = sns.kdeplot(blues[:,0], blues[:,1], cmap="Blues", shade=True, shade_lowest=False)

    # Add labels to the plot
    red = sns.color_palette("Reds")[-2]
    blue = sns.color_palette("Blues")[-2]
    ax.text(2.5, 8.2, "reds", size=16, color=blue)
    ax.text(3.8, 4.5, "blues", size=16, color=red)
    pyplot.show()


def user_input_test(model):
    while True:
        x = input('Enter x,y: ')
        if x:
            x = [[float(i) for i in x.split(',')]]
            x = np.array(x, dtype='float32')
            prediction = model.predict(x)
            print(prediction, '--> class:', prediction.argmax())
        else:
            break


def build_model():
    rn = initializers.glorot_uniform()
    inputs = Input(shape=(2,), dtype='float32')
    x = Dense(5, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(inputs)
    x = ReLU()(x)
    x = Dense(2, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(x)
    x = ReLU()(x)
    x = Dense(1, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(x)
    outputs = Activation('sigmoid')(x)
    return Model(inputs=inputs, outputs=outputs)


def xor_dataset(batches):
    data_shape = (batches,2)
    data = np.random.random(data_shape)
    bools = data < 0.5
    bools = bools[:,0] != bools[:,1]
    classes = np.ones((batches,1), dtype='float32')
    classes[:,0] = classes[:,0] * bools
    return data, classes


def jain_dataset():
    jain = np.loadtxt(path.join(__file__, '../clusters-jain.txt'), dtype='float32', delimiter='\t')
    x = jain[:,0:2]
    y = jain[:,2:3] - 1.0
    return x, y


def main():

    x, y = jain_dataset()

    bins, _ = np.histogram(y[:,0], bins=2)
    print('Histogram:', bins)

    controller = UserControlledLearningRate()

    start = time.time()

    def train(epochs):
        model = build_model()
        model.summary()
        opt = Adadelta()
        model.compile(optimizer=opt, loss=binary_crossentropy, metrics=['accuracy'])
        model.fit(x=x, y=y, epochs=epochs, validation_split=0.5, shuffle=True, verbose=2, batch_size=10, callbacks=[controller])
        return model

    with tf.device('/cpu:0'):
        model = train(10000)
        prediction = model.predict(x)

    print('\nElapsed', time.time() - start, 'seconds')
    draw_seaborn_scatter(x, prediction)
