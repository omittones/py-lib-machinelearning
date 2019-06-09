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

NMBALLS = 3


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
    x = keras.layers.Dense(100, use_bias=True)(x)
    x = keras.layers.ReLU(negative_slope=0)(x)
    x = keras.layers.Dense(50, use_bias=True, kernel_initializer='glorot_uniform')(x)
    x = keras.layers.ReLU(negative_slope=0)(x)
    x = keras.layers.Dense(nm_outputs, use_bias=True, kernel_initializer='glorot_uniform')(x)
    # x = keras.layers.ReLU(negative_slope=0.1)(x)
    # x = keras.layers.Activation(activation='tanh')(x)

    if softmax:
        x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=i, outputs=x)

    return model, loss


def process_observations(o0, o1, x, y, importance):
    assert len(o0) == NMBALLS * 6 and len(o1) == NMBALLS * 6
    for i in range(0, NMBALLS * 6, 6):
        inputs = list()
        inputs.extend(o0[i:i+6])
        inputs.extend(o0[:i])
        inputs.extend(o0[i+6:])
        outputs = o1[i:i+6]

        s0 = np.array(o0[i + 2: i + 4])
        s1 = np.array(o1[i + 2: i + 4])
        sd = np.linalg.norm(s0 - s1)
        x.append(np.array(inputs))
        y.append(np.array(outputs))
        importance.append(sd)


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
                process_observations(observation0, observation1, x, y, speed_diffs)

            if gui.IsCompleted: return

    model, loss = build_model(x[0], y[0], softmax=False)
    model.summary()

    nx = np.array(x, dtype='float32')
    ny = np.array(y, dtype='float32')
    nsd = np.array(speed_diffs, dtype='float32')
    print('Speed diff histogram:', np.histogram(nsd)[0])
    nsd = nsd.clip(0.001, 1)

    model.compile(optimizer=keras.optimizers.Adadelta(lr=1), loss=loss)
    model.fit(x=nx, y=ny, sample_weight=nsd, validation_split=0.2, verbose=1, epochs=400, batch_size=1000, shuffle=True, callbacks=[controller])

    def from_ball(ball):
        return [ball.Position.X, ball.Position.Y, ball.Speed.X, ball.Speed.Y, ball.Acceleration.X, ball.Acceleration.Y]

    while not gui.IsCompleted:

        env.Reset()
        inputs = np.zeros((NMBALLS, NMBALLS * 6))

        for e in range(0, 1000):
            for i in range(0, NMBALLS):
                ball = env.Objects[i]
                inputs[i, 0:6] = from_ball(ball)
                position = 6
                for j in range(0, NMBALLS):
                    if i != j:
                        inputs[i, position:position+6] = from_ball(env.Objects[j])
                        position += 6
            result = model.predict(inputs, batch_size=NMBALLS)
            result = result.clip(-1, 1)
            for i in range(0, NMBALLS):
                env.Objects[i].Position = PointF(result[i, 0], result[i, 1])
                env.Objects[i].Speed = PointF(result[i, 2], result[i, 3])
                env.Objects[i].Acceleration = PointF(result[i, 4], result[i, 5])
            time.sleep(.02)

    gui.Wait()
    if gui.Exception != None:
        print(gui.Exception)

if __name__ == '__main__':
    main()