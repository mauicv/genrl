"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from gerel.algorithms.RES.population import RESPopulation
from gerel.algorithms.RES.mutator import ADRESMutator
from gerel.populations.genome_seeders import curry_genome_seeder
import numpy as np
from gerel.genome.factories import dense
from gerel.util.datastore import DataStore
from examples.utils import build_env, make_counter_fn


def bipedal_walker_ADRES():
    genome = dense(
        input_size=24,
        output_size=4,
        layer_dims=[20]
    )

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

    mutator = ADRESMutator(
        initial_mu=init_mu,
        std_dev=0.5,
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=400,
        genome_seeder=seeder
    )

    ds = DataStore(name='bip_walker_ADRES_data')

    assign_population_fitness = build_env(
        env_name='BipedalWalker-v3',
        num_steps=200,
        repetition=1)
    counter_fn = make_counter_fn()

    for i in range(500):
        success = assign_population_fitness(population)
        if success and counter_fn():
            break
        data = population.to_dict()
        mutator(population)
        ds.save(data)
        print_progress(data)
    return True


def print_progress(data):
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)


if __name__ == '__main__':
    bipedal_walker_ADRES()
