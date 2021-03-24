"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src.algorithms.NEAT.population import NEATPopulation
from src.algorithms.NEAT.mutator import NEATMutator
from src import Model
from src import generate_neat_metric
from src import curry_genome_seeder
import gym
import numpy as np
from src.util import print_population
import matplotlib.pyplot as plt


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


def neat_cart_pole():
    pop_size = 1000
    mutator = NEATMutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    seed_genome = Genome.default(
        input_size=4,
        output_size=1,
        depth=5
    )
    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[seed_genome]
    )
    metric = generate_neat_metric(
        c_1=1,
        c_2=1,
        c_3=3
    )
    population = NEATPopulation(
        population_size=pop_size,
        genome_seeder=seeder,
        delta=4,
        metric=metric
    )

    train_data = []
    for i in range(5):
        for genome in population.genomes:
            reward = compute_n_fitness(5, genome.to_reduced_repr)
            genome.fitness = reward
        population.speciate()
        data = population.to_dict()
        print_progress(data)
        mutator(population)
        train_data.append(data["mean_fitness"])

    plt.plot(np.array(train_data))
    plt.show()
    print_population(population)
    score = compute_fitness(data['best_genome'], True)
    print(f'best_fitness: {score}')
    return True


def print_progress(data):
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)


if __name__ == "__main__":
    neat_cart_pole()
