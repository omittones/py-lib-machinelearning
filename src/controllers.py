from timeit import default_timer
from contextlib import contextmanager
from keras import backend as K
from keras.callbacks import Callback
from utils import key, readchar

@contextmanager
def elapsed_timer():
    start = default_timer()
    elapser = lambda: default_timer() - start
    yield lambda: elapser()
    end = default_timer()
    elapser = lambda: end-start

class Stopwatch():
    """Class used to measure elapsed time"""

    def __init__(self, text = None):
        self.start = default_timer()
        self.text = text

    def reset(self):
        self.start = default_timer()

    def checkpoint(self, text):
        seconds = None
        if (self.end):
            seconds = (default_timer() - self.start)
        else:
            seconds = (self.end - self.start)
        print(text, f"{seconds}s")

    def __enter__(self):
        self.start = default_timer()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.end = default_timer()
        if (self.text):
            self.checkpoint(self.text)


class UserControlledLearningRate(Callback):
    """
    Controller that accepts user input for controlling learning rate
    and stopping condition during training.
    """

    def __init__(self):
        super().__init__()
        self.rate = 0.1

    def set_model(self, model):
        super().set_model(model)
        if not hasattr(self.model.optimizer, 'lr'):
            raise ValueError('Optimizer must have a "lr" attribute.')
        self.rate = K.get_value(self.model.optimizer.lr)
        self.rate_changed = False
        print(f'Setting initial learning rate to {self.rate}')

    def handle_key(self):
        char = readchar()
        if char == key.UP:
            self.handle_up()
        elif char == key.DOWN:
            self.handle_down()
        elif char == key.ESC:
            self.handle_esc()

    def handle_esc(self):
        self.model.stop_training = True

    def handle_up(self):
        self.rate *= 2.0
        self.rate_changed = True
        print('\nScheduling rate change to', self.rate)

    def handle_down(self):
        self.rate /= 2.0
        self.rate_changed = True
        print('\nScheduling rate change to', self.rate)

    def on_batch_begin(self, batch, logs=None):
        self.handle_key()

    def on_epoch_begin(self, epoch, logs=None):
        if self.rate_changed:
            text = f'Epoch {epoch}: changed rate to {self.rate}'
            pad = '-' * len(text)
            print(pad)
            print(text)
            print(pad)
            K.set_value(self.model.optimizer.lr, self.rate)
            self.rate_changed = False

    def on_epoch_end(self, epoch, logs=None):
        pass