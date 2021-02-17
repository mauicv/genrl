"""Model.py

Transforms Genome object into a tensorflow graph structure.

inspired by: https://github.com/crisbodnar/TensorFlow-NEAT/blob/master/tf_neat/recurrent_net.py
"""

from tensorflow.keras.layers import Layer

class Linear(keras.layers.Layer):
    def __init__(self, genome):
        super(Linear, self).__init__()
        # self.w = self.add_weight(
        #     shape=(input_dim, units), initializer="random_normal", trainable=True
        # )
        # self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True)

    def call(self, inputs):
        return tf.matmul(inputs, self.w) + self.b
