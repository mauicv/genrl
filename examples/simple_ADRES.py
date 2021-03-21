"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src import RESPopulation
from src.RES.mutator import ADRESMutator
from src import curry_genome_seeder
import numpy as np


def simple_adres_example():
    genome = Genome.default(
        input_size=1,
        output_size=1
    )

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-3, 3, weights_len)

    mutator = ADRESMutator(
        initial_mu=init_mu,
        std_dev=0.1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=1000,
        genome_seeder=seeder
    )

    target_mu = np.random.uniform(-3, 3, len(init_mu))

    losses = []
    for i in range(100):
        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness

        mutator(population)
        loss = np.linalg.norm(mutator.mu - target_mu)
        losses.append(loss)
        print(f'generation: {i}, loss: {loss}')


if __name__ == '__main__':
    simple_adres_example()
