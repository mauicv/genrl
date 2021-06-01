"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from gerel.genome.factories import minimal
from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import ADRESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
import numpy as np
from examples.utils import build_simple_env


def simple_adres_example():
    genome = minimal(
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
    assign_population_fitness = build_simple_env(target_mu)

    for i in range(100):
        assign_population_fitness(population)
        mutator(population)
        loss = np.linalg.norm(mutator.mu - target_mu)
        print(f'generation: {i}, loss: {loss}')


if __name__ == '__main__':
    simple_adres_example()
