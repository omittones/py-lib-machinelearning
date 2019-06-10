import time
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from os import path
import io
import keras
from keras import activations, initializers
from keras.layers import Activation, Dense, Input, Softmax, ReLU, BatchNormalization
from keras.models import Model
from keras.optimizers import SGD, Adadelta
from controllers import UserControlledLearningRate
import tensorflow as tf


def draw_seaborn_scatter(data, prediction):
    sns.set(style="darkgrid")
    hue = prediction.argmax(axis=1)
    p = sns.color_palette("Paired", prediction.shape[1])
    p = p[0:hue.max() + 1] #bug when more colors than categories
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=hue, palette=p)
    plt.show()


def draw_seaborn_density(data, prediction):

    reds = data[prediction[:,0]>0.5,:]
    blues = data[prediction[:,1]>0.5,:]

    sns.set(style="darkgrid")
    _, ax = plt.subplots(figsize=(6, 6))
    ax.set_aspect("equal")

    # Draw the two density plots
    ax = sns.kdeplot(reds[:,0], reds[:,1], cmap="Reds", shade=True, shade_lowest=False)
    ax = sns.kdeplot(blues[:,0], blues[:,1], cmap="Blues", shade=True, shade_lowest=False)

    # Add labels to the plot
    red = sns.color_palette("Reds")[-2]
    blue = sns.color_palette("Blues")[-2]
    ax.text(2.5, 8.2, "reds", size=16, color=blue)
    ax.text(3.8, 4.5, "blues", size=16, color=red)
    plt.show()


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


def s1_dataset():
    x = np.loadtxt(path.join(__file__, '../cluster-s1-data.txt'), dtype='float32', delimiter='    ')
    labels = np.loadtxt(path.join(__file__, '../cluster-s1-labels.txt'), dtype='int8')
    x = x - x.min()
    x = x / x.max()

    labels = labels - labels.min()
    max_label = labels.max()
    y = np.zeros((labels.shape[0], max_label + 1), dtype='float32')
    for i,l in enumerate(labels):
        y[i,l] = 1

    shuffled = np.random.permutation(x.shape[0])
    return x[shuffled], y[shuffled]


def build_model(hidden_node_count = 4):
    rn = initializers.glorot_uniform()
    inputs = Input(shape=(2,), dtype='float32')
    x = Dense(hidden_node_count, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(inputs)
    x = ReLU()(x)
    x = Dense(hidden_node_count, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(x)
    x = ReLU()(x)
    x = Dense(15, use_bias=True, kernel_initializer=rn, bias_initializer='zeros')(x)
    outputs = Softmax()(x)
    return Model(inputs=inputs, outputs=outputs)


def main():

    x, y = s1_dataset()
    # draw_seaborn_scatter(x, y)

    bins, _ = np.histogram(y.argmax(axis=1))
    print('Histogram:', bins)

    controller = UserControlledLearningRate()

    start = time.time()

    def train(model, epochs):
        opt = Adadelta()
        model.compile(optimizer=opt, loss=keras.losses.categorical_crossentropy, metrics=['accuracy'])
        history = model.fit(x=x, y=y, epochs=epochs, validation_split=0.2, shuffle=True, verbose=2, batch_size=1000, callbacks=[controller])

        # plt.plot(history.history['acc'])
        # plt.plot(history.history['val_acc'])
        # plt.title('model accuracy')
        # plt.ylabel('accuracy')
        # plt.xlabel('epoch')
        # plt.legend(['train', 'test'], loc='upper left')
        # plt.show()
        # summarize history for loss
        plt.plot(history.history['loss'])

        return model

    labels = []
    with tf.device('/gpu:0'):
        hnc = 5
        labels.append(f'comp-{hnc}')
        model = build_model(hnc)
        model.summary()
        model = train(model, 2000)
        test = np.random.rand(5000, 2) * (x.max() - x.min()) + x.min()
        test = test.astype('float32')
        prediction = model.predict(test)
        print('\nDone', labels[-1])
        print('\nElapsed', time.time() - start, 'seconds')

        plt.title('model loss')
        plt.ylabel('loss')
        plt.xlabel('epoch')
        plt.legend(labels, loc='upper left')
        draw_seaborn_scatter(test, prediction)
