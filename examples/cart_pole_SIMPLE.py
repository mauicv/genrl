"""Example Implementation of SIMPLE-ES algorithm from"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from gerel.genome.factories import dense
from gerel.algorithms.SIMPLE.population import SIMPLEPopulation
from gerel.algorithms.SIMPLE.mutator import SIMPLEMutator
from gerel.populations.genome_seeders import curry_genome_seeder
from examples.utils import build_env, run_env, make_counter_fn


def cart_pole_simple_example():
    genome = dense(
        input_size=4,
        output_size=1,
        layer_dims=[2, 2, 2]
    )

    mutator = SIMPLEMutator(
        std_dev=0.1,
        survival_rate=0.1
    )

    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[genome]
    )

    population = SIMPLEPopulation(
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
    run_env(data['best_genome'])


if __name__ == '__main__':
    cart_pole_simple_example()
