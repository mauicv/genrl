"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src.genome.factories import dense
from src import RESPopulation
from src.RES.mutator import ADRESMutator
from src import Model
from src import curry_genome_seeder
import gym
import numpy as np


def compute_fitness(genome, render=False):
    model = Model(genome)
    env = gym.make("CartPole-v0")
    state = env.reset()
    fitness = 0
    action_map = lambda a: 0 if a[0] <= 0 else 1
    for _ in range(1000):
        action = model(state)
        action = action_map(action)
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


def cart_pole_res_example():
    genome = dense(
        input_size=24,
        output_size=4,
        layer_dims=[20]
    )


    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

    mutator = ADRESMutator(
        initial_mu=init_mu,
        std_dev=0.1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=50,
        genome_seeder=seeder
    )

    for i in range(100):
        fitness = 0
        for genome in population.genomes:
            fitness = compute_n_fitness(3, genome.to_reduced_repr)
            genome.fitness = fitness
        if fitness == 200:
            break
        data = population.to_dict()
        mutator(population)
        print(f'generation: {i}, mean score: {data["mean_fitness"]}, best score: {data["best_fitness"]}')

    data = population.to_dict()
    compute_fitness(data['best_genome'], render=True)


if __name__ == '__main__':
    cart_pole_res_example()
