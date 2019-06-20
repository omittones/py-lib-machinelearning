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
from os import path

NMBALLS = 1


def from_ball(ball):
    return [ball.Position.X, ball.Position.Y, ball.Speed.X, ball.Speed.Y] #, ball.Acceleration.X, ball.Acceleration.Y]


def to_ball(ball, values):
    ball.Position = PointF(values[0], values[1])
    ball.Speed = PointF(values[2], values[3])
    # ball.Acceleration = PointF(values[4], values[5])


def process_observations(o0, o1, i):
    assert len(o0) == NMBALLS * 6 and len(o1) == NMBALLS * 6
    i = i * 6
    inputs = list()
    inputs.extend(o0[i:i+4]) # position and speed for current ball
    outputs = o1[i:i+4] # new position and speed
    for j in range(0, NMBALLS * 6, 6): # positions of other balls
        if j != i:
            inputs.extend(o0[j:j+2])
    inputs = np.array(inputs)
    outputs = np.array(outputs)

    s0 = inputs[2:4]
    s1 = outputs[2:4]
    sd = np.linalg.norm(s0 - s1) # change in speed dictates importance

    return inputs, outputs, sd

def build_model(input_shape, output_shape, softmax=False):

    gu = keras.initializers.glorot_uniform(12346)

    if softmax:
        loss = keras.losses.categorical_crossentropy
    else:
        loss = keras.losses.mean_squared_error

    i = keras.layers.Input(shape=input_shape, dtype='float32')

    x = i
    x = keras.layers.Dense(20, use_bias=True, kernel_initializer=gu)(x)
    x = keras.layers.ReLU()(x)
    x = keras.layers.Dense(np.prod(output_shape), use_bias=True, kernel_initializer=gu)(x)

    if softmax:
        x = keras.layers.Softmax()(x)

    model = keras.Model(inputs=i, outputs=x)

    return model, loss


def run_sim_and_return_movement(env):
    actions = np.zeros((NMBALLS, 5), dtype='float32')
    actions[:,0] = 1
    actions = actions.flatten()
    x = list()
    y = list()
    speed_diffs = list()
    observations = list()

    for e in range(0, 100):
        env.Reset()
        observations.append(None)
        for i in range(0, 4001):
            r = env.Step(actions)
            observations.append(list(r.Observation))

    for (obs0, obs1) in zip(observations[:-1], observations[1:]):
        if obs0 is not None and obs1 is not None:
            for b in range(0, NMBALLS):
                sx, sy, sw = process_observations(obs0, obs1, b)
                x.append(sx)
                y.append(sy)
                speed_diffs.append(sw)

    nx = np.array(x, dtype='float32')
    ny = np.array(y, dtype='float32')
    nsd = np.array(speed_diffs, dtype='float32')
    return nx, ny, nsd


def main():

    env = Environment(NMBALLS)
    env.FrictionFactor = 1

    def create_view():
        form = EnvironmentDisplay()
        form.Renderer = Renderer(env)
        return form

    gui = GUI.ShowForm(Func[Form](create_view))

    model = None

    class Stopper(keras.callbacks.Callback):
        def on_batch_begin(self, batch, logs=None):
            char = readchar()
            if char == key.ESC or gui.IsCompleted:
                self.model.stop_training = True
    controller = Stopper()
    controller = UserControlledLearningRate()

    movement_file = path.join(__file__, f'../movement_for_{NMBALLS}_balls.npz')
    try:
        file = np.load(movement_file)
        nx = file['nx']
        ny = file['ny']
        nsd = file['nsd']
        file.close()
    except:
        nx, ny, nsd = run_sim_and_return_movement(env)
        np.savez(movement_file, nx = nx, ny = ny, nsd = nsd)

    nx[:,2:4] = nx[:,2:4] * 0.02
    ny[:,2:4] = ny[:,2:4] * 0.02

    # nx = nx[nsd <= 0]
    # ny = ny[nsd <= 0]
    # nsd = nsd[nsd <= 0]
    # rx = np.random.randint(0, nx.shape[0], dtype='int32')
    # print('X:', nx[rx])
    # print('Y:', ny[rx])
    # print('SW:', nsd[rx])
    # print('EY:', nx[rx, 0:2] + nx[rx, 2:4])
    # return

    # SUM = 100000.0
    # h, b = np.histogram(nsd)
    # print('Speed diff histogram:', h)
    # h = SUM / h.size / h
    # nsd = np.digitize(nsd, b[1:], right=True)
    # nsd = h[nsd]
    # assert nsd.sum() < SUM + 0.1 and nsd.sum() > SUM - 0.1

    model, loss = build_model(nx.shape[1:], ny.shape[1:], softmax=False)
    model.summary()

    filename = path.join(__file__, '../learn_physics.h5')
    if path.exists(filename):
        print(f'Loading weights from {filename}')
        model.load_weights(filename)

    optimizer = keras.optimizers.Adadelta(lr=1)
    optimizer = keras.optimizers.SGD(lr=0.5)
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(x=nx, y=ny,
        #sample_weight=nsd,
        validation_split=0.2,
        verbose=2,
        epochs=500,
        batch_size=20000,
        shuffle=True,
        callbacks=[controller])

    print(f'\nSaving weights to {filename}')
    model.save_weights(filename)

    while not gui.IsCompleted:

        env.Reset()
        inputs = np.zeros((NMBALLS, 2 + NMBALLS * 2))

        for e in range(0, 1000):
            for i in range(0, NMBALLS):
                ball = env.Objects[i]
                inputs[i, 0:4] = from_ball(ball)
                position = 4
                for j in range(0, NMBALLS):
                    if i != j:
                        inputs[i, position] = env.Objects[j].Position.X
                        inputs[i, position + 1] = env.Objects[j].Position.Y
                        position += 2

            inputs[:, 2:4] = inputs[:, 2:4] * 0.02
            result = model.predict(inputs, batch_size=NMBALLS)
            result[:, 2:4] = result[:, 2:4] / 0.02

            for i in range(0, NMBALLS):
                to_ball(env.Objects[i], result[i])

    gui.Wait()
    if gui.Exception != None:
        print(gui.Exception)

if __name__ == '__main__':
    main()