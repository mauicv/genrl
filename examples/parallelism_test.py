import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

import gym
import time
from src import BatchJob


def cart_pole(data):
    for _ in data:
        env = gym.make("CartPole-v0")
        env.reset()
        for _ in range(1000):
            env.step(env.action_space.sample())
    return True


def test_parallelism():
    batch_job = BatchJob(num_processes=4)
    batch_cart_pole = batch_job(cart_pole)

    start = time.time()
    [cart_pole([None]) for _ in range(100)]
    end = time.time()
    duration_1 = end - start
    print('Time in serial:', round(duration_1, 3))

    start = time.time()
    batch_cart_pole([[None] for _ in range(100)])
    end = time.time()
    duration_2 = end - start
    print('Time in parallel:', round(duration_2, 3))

    print('Ratio of Processing Times:', round(duration_1/duration_2, 3))
    return True
