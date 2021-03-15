"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src import NEATPopulation
from src import NEATMutator
from src import Model
from src import generate_neat_metric
import gym
import numpy as np
from src.util import save


def compute_fitness(genome, render=False):
    model = Model(genome)
    env = gym.make("BipedalWalker-v3")
    state = env.reset()
    fitness = 0
    for _ in range(200):
        action = model(state)
        action = np.tanh(np.array(action))
        state, reward, done, _ = env.step(action)
        fitness += reward
        if render:
            env.render()

        if done:
            break

    return fitness


def compute_n_fitness(n, genome):
    fitness = 0
    for i in range(n):
        fitness += compute_fitness(genome)
    return fitness/n


def test_bip_walker():
    pop_size = 400
    m = NEATMutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    g = Genome.default(
        input_size=24,
        output_size=4,
        depth=5
    )
    p = NEATPopulation(
        population_size=pop_size,
        seed_genomes=[g],
        mutator=m,
        delta=4
    )
    d = generate_neat_metric(
        c_1=1,
        c_2=1,
        c_3=3
    )

    save(fname='save.txt', data=None)

    for i in range(3):
        for g in p.genomes:
            reward = compute_n_fitness(1, g.to_reduced_repr)
            g.fitness = reward
        p.speciate(metric=d)
        data = p.to_dict()
        p.evolve()
        print_progress(data)
        save(data, 'save.txt')
    return True


def print_progress(data):
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)

