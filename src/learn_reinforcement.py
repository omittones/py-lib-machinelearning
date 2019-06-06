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
from controllers import StopOnEscape


def build_model(nm_inputs, nm_outputs, softmax=False):
    if softmax:
        loss = keras.losses.categorical_crossentropy
    else:
        loss = keras.losses.mean_squared_error

    i = keras.layers.Input(shape=(nm_inputs,), dtype='float32')

    x = i
    # x = keras.layers.Reshape((nm_inputs, 1,))(x)
    # x = keras.layers.Conv1D(30, 6, strides=6, padding='valid')(x)
    # x = keras.layers.ReLU()(x)
    # x = keras.layers.Reshape((150, 1))(x)
    # x = keras.layers.Conv1D(30, 30, strides=30, padding='valid')(x)
    # x = keras.layers.Reshape((150, 1))(x)
    # x = keras.layers.Conv1D(30, 30, strides=30, padding='valid')(x)
    # x = keras.layers.ReLU()(x)
    # x = keras.layers.Flatten()(x)

    x = keras.layers.Dense(nm_inputs * 30)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(nm_inputs * 20)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(nm_inputs * 10)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(nm_outputs)(x)

    if softmax:
        x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=i, outputs=x)
    model.compile(optimizer=keras.optimizers.Adadelta(), loss=loss)

    return model


def main():

    env = Environment()
    env.FrictionFactor = 1

    def create_view():
        form = EnvironmentDisplay()
        form.Renderer = Renderer(env)
        return form

    gui = GUI.ShowForm(Func[Form](create_view))

    env.Reset()
    x = list()
    y = list()
    observation0 = None
    observation1 = None
    model = None

    class Stopper(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs=None):
            char = readchar()
            if char == key.ESC or gui.IsCompleted:
                self.model.stop_training = True
    controller = Stopper()

    while not gui.IsCompleted:
        for i in range(0, 20001):
            observation0 = observation1
            for s in range(0, 10):
                r = env.Step(
                    [1,0,0,0,0,
                    1,0,0,0,0,
                    1,0,0,0,0,
                    1,0,0,0,0,
                    1,0,0,0,0])
            observation1 = list(r.Observation)
            if model is None:
                model = build_model(len(observation1), len(observation1), softmax=False)
                model.summary()
            if observation0 is not None:
                x.append(observation0)
                y.append(observation1)

        nx = np.array(x, dtype='float32')
        ny = np.array(y, dtype='float32')
        model.fit(x=nx, y=ny, verbose=1, epochs=100, batch_size=100, shuffle=True, callbacks=[controller])
        break

    state0 = np.array(observation1).reshape((1,30,))
    while not gui.IsCompleted:
        state1 = model.predict(state0, batch_size=1)
        for i in range(0, env.Objects.Length):
            x = i * 6
            env.Objects[i].Position = PointF(state1[0, x], state1[0, x+1])
            env.Objects[i].Speed = PointF(state1[0, x+2], state1[0, x+3])
            env.Objects[i].Acceleration = PointF(state1[0, x+4], state1[0, x+5])
        state0 = state1
        time.sleep(.1)

    gui.Wait()
    if gui.Exception != None:
        print(gui.Exception)

if __name__ == '__main__':
    main()