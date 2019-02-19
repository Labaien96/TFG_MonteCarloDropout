import numpy as np
import tensorflow as tf
import keras.backend as K


class Dropout_permanent(Layer):
    def __init__(self, rate, **kwargs):
        super(Dropout_permanent, self).__init__(**kwargs)
        self.rate = min(1., max(0., rate))
        self.supports_masking = True

    def call(self, inputs, training=None):
        if 0. < self.rate < 1.:
            retain_prob = 1. - self.rate

            def dropped_inputs():
                return tf.nn.dropout(inputs, retain_prob, None, seed=np.random.randint(10e6))

            return K.in_train_phase(dropped_inputs, dropped_inputs, training=training)

        return inputs

    def get_config(self):
        config = {'rate': self.rate}
        base_config = super(Dropout_permanent, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))

    def compute_output_shape(self, input_shape):
        return input_shape