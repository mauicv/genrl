"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src import Population
from src import Mutator
from src import Model
from src import generate_neat_metric
from tqdm import tqdm
import gym
import numpy as np
from src.batch import BatchJob
from src.util import print_population, save, load


def play(genome):
    model = Model(genome)
    env = gym.make("BipedalWalker-v3")
    state = env.reset()
    fitness = 0
    done = False
    while not done:
        action = model(state)
        action = np.tanh(np.array(action))
        state, reward, done, _ = env.step(action)
        fitness += reward
        env.render()

    return fitness


def play_bip_walker():
    data = load('save.txt')
    play(data)
    return True
