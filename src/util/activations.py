import numpy as np
from math import exp

def step(x):
    return 1 if x > 0 else - 1

def sigmoid(x, c=4.9):
    return 1/(1+exp(-c*x))
