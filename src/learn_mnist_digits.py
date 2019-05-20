from controllers import UserControlledLearningRate
from keras import initializers
import time
from keras.datasets import mnist
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.models import Model
from keras.layers import Input, Dense, LeakyReLU, Softmax, Reshape
from keras.losses import categorical_crossentropy
import seaborn as sns
import matplotlib.pyplot as plt

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

def main():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    x_train, y_train = prepare_data(x_train, y_train)
    x_test, y_test = prepare_data(x_test, y_test)

    # figure = plt.figure()
    # for i in range(0, 100):
    #     ax = figure.add_subplot(10, 10, i + 1)
    #     ax.imshow(x_train[i], cmap='gray'), plt.axis("off")
    # plt.show()

    model = build_model()
    controller = UserControlledLearningRate()

    def train(epochs, size, batch_size):
        x = x_train[0:size]
        y = y_train[0:size]
        sgd = SGD(lr=0.5, momentum=0.01, decay=0.0, nesterov=False)
        model.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])
        model.fit(x=x, y=y, epochs=epochs, verbose=2, batch_size=batch_size, callbacks=[controller], shuffle=True)

    start = time.time()


    # train(100, 100, 10)
    # train(100, 1000, 100)
    # train(100, 10000, 1000)
    # train(100, 60000, 6000)
    # Elapsed 78.5720272064209 seconds
    # Test error rate: 4.250001907348633 %

    # train(80, 60000, 100)
    # Elapsed 77.50399446487427 seconds
    # Test error rate: 2.0299971103668213 %

    train(500, 60000, 100)
    print('Elapsed', time.time() - start, 'seconds')

    final = model.test_on_batch(x_test, y_test)
    final = dict(zip(model.metrics_names, final))
    print('Test error rate:', (1 - final['acc']) * 100, '%')