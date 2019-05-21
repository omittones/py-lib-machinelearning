import keyboard
from keras import backend as K
from keras.callbacks import Callback

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
        keyboard.on_press_key('esc', self.handle_esc)
        keyboard.on_press_key(keyboard.KEY_UP, self.handle_up)
        keyboard.on_press_key(keyboard.KEY_DOWN, self.handle_down)

    def handle_esc(self, e):
        self.model.stop_training = True

    def handle_up(self, e):
        self.rate *= 2.0
        self.rate_changed = True

    def handle_down(self, e):
        self.rate /= 2.0
        self.rate_changed = True

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