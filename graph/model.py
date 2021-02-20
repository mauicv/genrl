"""Model.py

Transforms Genome object into a tensorflow graph structure.

inspired by: https://github.com/crisbodnar/TensorFlow-NEAT/blob/master/tf_neat/recurrent_net.py
""" # noqa

#
# from tensorflow.keras.layers import Layer
#
# class Linear(keras.layers.Layer):
#     def __init__(self, genome):
#         super(Linear, self).__init__()
#         self.w = self.add_weight(
#           shape=(input_dim, units), initializer="random_normal", trainable=True # noqa
#         )
#         self.b = self.add_weight(shape=(units,), initializer="zeros", trainable=True) # noqa
#
#     def call(self, inputs):
#         return tf.matmul(inputs, self.w) + self.b
