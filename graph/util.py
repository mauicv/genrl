import random


def sample_weight(low, high):
    range = high - low
    return random.random() * range + low


def get_random():
    return random.random() * 2 - 1
