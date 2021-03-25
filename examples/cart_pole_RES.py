"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src.genome.factories import dense
from src.algorithms.RES.population import RESPopulation
from src.algorithms.RES.mutator import RESMutator
from src import curry_genome_seeder
import numpy as np
from examples.utils import build_env, run_env, make_counter_fn


def cart_pole_res_example():
    genome = dense(
        input_size=4,
        output_size=1,
        layer_dims=[2, 2, 2]
    )

    weights_len = len(genome.edges) + len(genome.nodes)
    init_mu = np.random.uniform(-1, 1, weights_len)

    mutator = RESMutator(
        initial_mu=init_mu,
        std_dev=0.1,
        alpha=0.05
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = RESPopulation(
        population_size=50,
        genome_seeder=seeder
    )

    assign_population_fitness = build_env()
    counter_fn = make_counter_fn()

    for i in range(100):
        success = assign_population_fitness(population)
        if success and counter_fn():
            break
        data = population.to_dict()
        mutator(population)
        print(f'generation: {i}, mean score: {data["mean_fitness"]}, best score: {data["best_fitness"]}')
    data = population.to_dict()
    run_env(data['best_genome'], env_name='CartPole-v0', render=True)


if __name__ == '__main__':
    cart_pole_res_example()
