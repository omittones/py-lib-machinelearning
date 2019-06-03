import keras.backend as K
from keras.optimizers import Optimizer
from math import sqrt
import numpy as np


class SlicingOptimizer(Optimizer):

    def __init__(self, lr=1, **kwargs):
        super().__init__(**kwargs)
        with K.name_scope(self.__class__.__name__):
            self.lr = K.variable(lr, name='lr')
            self.old_loss = K.variable(0, name='old_loss')

    def get_updates(self, loss, params):

        shapes = [K.int_shape(p) for p in params]
        movement = [np.random.randn(*s) for s in shapes]
        norm = sum([(m * m).sum() for m in movement])
        norm = sqrt(norm)
        for m in movement:
            m = m / norm

        self.updates = []
        for p, m in zip(params, movement):
            cm = K.constant(m, name='movement')
            self.updates.append(K.update_add(p, cm * self.lr))
        self.updates.append(K.update(self.old_loss, loss))
        return self.updates

    def get_config(self):
        config = {
                #   'lr': float(K.get_value(self.lr)),
                #   'rho': self.rho,
                #   'decay': float(K.get_value(self.decay)),
                #   'epsilon': self.epsilon
                }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))
