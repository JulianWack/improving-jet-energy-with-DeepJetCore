import tensorflow as tf
import tensorflow.keras as keras

#from caloGraphNN import euclidean_squared, gauss, gauss_of_lin

class GlobalExchange(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(GlobalExchange, self).__init__(**kwargs)

    def build(self, input_shape):
        # tf.ragged FIXME?
        self.num_vertices = input_shape[1]
        super(GlobalExchange, self).build(input_shape)

    def call(self, x):
        mean = tf.reduce_mean(x, axis=1, keepdims=True)
        # tf.ragged FIXME?
        # maybe just use tf.shape(x)[1] instead?
        mean = tf.tile(mean, [1, self.num_vertices, 1])
        return tf.concat([x, mean], axis=-1)

    def compute_output_shape(self, input_shape):
        return input_shape[:2] + (input_shape[2] * 2,)
