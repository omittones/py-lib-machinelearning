from keras.layers import Input, Dense, Softmax, LeakyReLU
from keras.models import Model
from keras.losses import categorical_crossentropy
from keras.optimizers import SGD
from matplotlib import pyplot
import numpy as np

def main():

    inputs = Input(shape=(2,), dtype='float32')
    x = Dense(10)(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(10)(inputs)
    x = LeakyReLU(alpha=0.3)(x)
    x = Dense(2)(x)
    outputs = Softmax()(x)

    sgd = SGD(lr=0.05, momentum=0.0, decay=0.0, nesterov=False)

    model = Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=sgd, loss=categorical_crossentropy, metrics=['accuracy'])

    data = np.random.random((1000, 2))
    classes = np.zeros((1000, 2))
    for i in range(0, 1000):
        x = data[i,0] < 0.5
        y = data[i,1] < 0.5
        classes[i, 0] = 0 if x != y else 1
        classes[i, 1] = 1 if x != y else 0

    model.fit(x=data, y=classes, epochs=50, verbose=2, batch_size=10)

    prediction = model.predict(data)

    # select 1st category and join it with data to get scatterplot
    cat1st = prediction[:,0].copy()
    cat1st.resize((1000, 1))
    graph = np.hstack([data, cat1st])

    # show graph
    pyplot.imshow(graph, cmap='hot', interpolation='nearest')
    pyplot.show()

    # test
    while True:
        x = input('Enter x,y: ')
        if x:
            x = [[float(i) for i in x.split(',')]]
            x = np.array(x, dtype='float32')
            prediction = model.predict(x)
            print(prediction, '--> class:', prediction.argmax())
        else:
            break