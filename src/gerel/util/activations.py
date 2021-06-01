import numpy as np
from math import exp


def step(x):
    return 1 if x > 0 else - 1


def relu(x):
    return max(0, x)


def tanh(x):
    return np.tanh(x)


def build_sigmoid(c=4.9):
    def sigmoid(x):
        return 1 / (1 + exp(-c * x))
    return sigmoid


def build_leaky_relu(c=0.01):
    def leaky_relu(x):
        return c * x if x < 0 else x
    return leaky_relu
