# pylint: skip-file

import clr
clr.AddReference("System.Windows.Forms")
clr.AddReference(r"D:\Code\cs-lib-machine-learning\Environments\bin\Debug\Environments.exe")

from System import Action, Func
from System.Threading.Tasks import Task
from Environments import GUI
from Environments.Bouncies import Environment, Renderer
from Environments.Forms import EnvironmentDisplay
from System.Drawing import PointF
from System.Windows.Forms import Form

import time
import keras
import numpy as np
from utils import key, readchar
from controllers import UserControlledLearningRate

NMBALLS = 1

def enhance_input(x):
    d = np.array(x)
    d = np.resize(d, (NMBALLS, 6))
    return d


def build_model(input_sample, output_sample, softmax=False):
    nm_inputs = input_sample.shape
    nm_outputs = len(output_sample)

    if softmax:
        loss = keras.losses.categorical_crossentropy
    else:
        loss = keras.losses.mean_squared_error

    i = keras.layers.Input(shape=nm_inputs, dtype='float32')

    def activation(t):
        return

    x = i
    x = keras.layers.Reshape((NMBALLS,6,1,))(x)
    # x = keras.layers.Conv2D(100, (1, 6), strides=(1,1), padding='valid', use_bias=True)(x)
    x = keras.layers.Flatten()(x)
    # x = keras.layers.ReLU(negative_slope=0.1)(x)
    # x = keras.layers.Dense(600, use_bias=True)(x)
    # x = keras.layers.ReLU(negative_slope=0.1)(x)
    x = keras.layers.Dense(50, use_bias=True, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.ReLU(negative_slope=0.1)(x)
    x = keras.layers.Dense(nm_outputs, use_bias=True, kernel_initializer='glorot_uniform')(x)
    # x = keras.layers.ReLU(negative_slope=0.1)(x)
    # x = keras.layers.Activation(activation='tanh')(x)

    if softmax:
        x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=i, outputs=x)

    return model, loss


def speed_diff(s0, s1):
    assert len(s0) == NMBALLS * 6 and len(s1) == NMBALLS * 6
    s = 0
    for i in range(0, NMBALLS):
        v0 = np.array(s0[i * 6 + 2: i * 6 + 4])
        v1 = np.array(s1[i * 6 + 2: i * 6 + 4])
        s += np.linalg.norm(v0 - v1)
    return s


def main():

    env = Environment(NMBALLS)
    env.FrictionFactor = 1

    def create_view():
        form = EnvironmentDisplay()
        form.Renderer = Renderer(env)
        return form

    gui = GUI.ShowForm(Func[Form](create_view))

    x = list()
    y = list()
    model = None

    class Stopper(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs=None):
            char = readchar()
            if char == key.ESC or gui.IsCompleted:
                self.model.stop_training = True
    controller = Stopper()

    speed_diffs = list()

    actions = np.zeros((NMBALLS, 5), dtype='float32')
    actions[:,0] = 1
    actions = actions.flatten()

    for e in range(0, 500):
        env.Reset()
        observation0 = None
        observation1 = None
        for i in range(0, 201):
            observation0 = observation1
            for s in range(0, 3):
                r = env.Step(actions)
            observation1 = list(r.Observation)

            if observation0 is not None:
                speed_diffs.append(speed_diff(observation0, observation1))
                x.append(enhance_input(observation0))
                y.append(observation1)

            if gui.IsCompleted: return

    model, loss = build_model(x[0], y[0], softmax=False)
    model.summary()

    nx = np.array(x, dtype='float32')
    ny = np.array(y, dtype='float32')
    nsd = np.array(speed_diffs, dtype='float32')
    print('Speed diff histogram:', np.histogram(nsd)[0])
    nsd = nsd.clip(0.05, 5)

    model.compile(optimizer=keras.optimizers.Adadelta(lr=5), loss=loss)
    model.fit(x=nx, y=ny, sample_weight=nsd, validation_split=0.2, verbose=2, epochs=400, batch_size=1000, shuffle=True, callbacks=[controller])

    state0 = observation1
    while not gui.IsCompleted:
        state0 = enhance_input(state0)
        state0 = np.expand_dims(state0, 0)
        state1 = model.predict(state0, batch_size=1)
        state1 = state1.clip(-1, 1)
        for i in range(0, env.Objects.Length):
            x = i * 6
            env.Objects[i].Position = PointF(state1[0, x], state1[0, x+1])
            env.Objects[i].Speed = PointF(state1[0, x+2], state1[0, x+3])
            env.Objects[i].Acceleration = PointF(state1[0, x+4], state1[0, x+5])
        time.sleep(.05)
        state0 = state1

    gui.Wait()
    if gui.Exception != None:
        print(gui.Exception)

if __name__ == '__main__':
    main()