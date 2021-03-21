"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src.RES.population import RESPopulation
from src.RES.mutator import RESMutator
from src import Model
from src import curry_genome_seeder
import gym
import numpy as np
from src.genome.factories import dense


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


def bipedal_walker_RES():
    genome = dense(
        input_size=24,
        output_size=4,
        layer_dims=[20]
    )

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=0.5,
        alpha=1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=400,
        genome_seeder=seeder
    )

    for i in range(20):
        for g in population.genomes:
            reward = compute_n_fitness(1, g.to_reduced_repr)
            g.fitness = reward
        data = population.to_dict()
        mutator(population)
        print_progress(data)
    compute_fitness(data['best_genome'], render=True)
    return True


def print_progress(data):
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)


if __name__ == '__main__':
    bipedal_walker_RES()
