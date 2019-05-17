from keras import initializers
from keras.layers import Input, Dense, Softmax, LeakyReLU
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from keras.callbacks import Callback, LearningRateScheduler
from keras import backend as K
import keyboard
import time
import numpy as np
import seaborn as sns
from matplotlib import pyplot

def draw_seaborn_scatter(data, prediction):
    sns.set(style="darkgrid")
    
    p = sns.blend_palette(['#ff0000','#ff0000','#0000ff','#0000ff'], as_cmap=True)
    f, ax = pyplot.subplots(figsize=(6, 6))
    ax.set_aspect("equal")
    sns.scatterplot(x=data[:,0], y=data[:,1], hue=prediction[:,0], palette=p)
    pyplot.show()


def draw_seaborn_density(data, prediction):
    
    reds = data[prediction[:,0]>0.5,:]
    blues = data[prediction[:,1]>0.5,:]

    sns.set(style="darkgrid")
    f, ax = pyplot.subplots(figsize=(6, 6))
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

class UserControlledLearningRate(Callback):

    def __init__(self):
        super().__init__()
        self.rate = 0.1

    def set_model(self, model):
        super().set_model(model)
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.rate = K.get_value(self.model.optimizer.lr)
        print(f'Setting initial learning rate to {self.rate}')

    def on_epoch_begin(self, epoch, logs=None):
        if keyboard.is_pressed(keyboard.KEY_UP):
            self.rate *= 2.0
        elif keyboard.is_pressed(keyboard.KEY_DOWN):
            self.rate /= 2.0
        else:
            return
        text = f'Epoch {epoch}: changed rate to {self.rate}' 
        pad = '-' * len(text)
        print(pad)
        print(text)
        print(pad)
        K.set_value(self.model.optimizer.lr, self.rate)

    def on_epoch_end(self, epoch, logs=None):
        pass

def main():
    
    data_shape = (1000,2)
    data = np.random.random(data_shape)
    bools = data < 0.5
    bools = bools[:,0] != bools[:,1]
    classes = np.ones(data_shape, dtype='float32')
    classes[:,0] = classes[:,0] * bools
    classes[:,1] = classes[:,1] * np.invert(bools)  
    
    rn = initializers.RandomNormal(mean=0.0, stddev=0.005, seed=4337)

    inputs = Input(shape=(2,), dtype='float32')
    x = Dense(10, kernel_initializer=rn, bias_initializer=rn)(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(5, kernel_initializer=rn, bias_initializer=rn)(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(2, kernel_initializer=rn, bias_initializer=rn)(x)
    outputs = Softmax()(x)

    model = Model(inputs=inputs, outputs=outputs)

    start = time.time()
    
    controller = UserControlledLearningRate()
    
    def train(epochs):
        sgd = SGD(lr=0.1, momentum=0.0, decay=0.0, nesterov=False)
        model.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])
        model.fit(x=data, y=classes, epochs=epochs, verbose=2, batch_size=10, callbacks=[controller])

    train(300)

    prediction = model.predict(data)
    print('Elapsed', time.time() - start, 'seconds')
    draw_seaborn_scatter(data, prediction)