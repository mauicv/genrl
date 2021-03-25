"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src.genome.factories import minimal
from src.algorithms.NEAT.population import NEATPopulation
from src.algorithms.NEAT.mutator import NEATMutator
from src import generate_neat_metric
from src import curry_genome_seeder
from examples.utils import build_env, run_env, make_counter_fn


def neat_cart_pole():
    pop_size = 1000
    mutator = NEATMutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    seed_genome = minimal(
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

    assign_population_fitness = build_env()
    counter_fn = make_counter_fn()

    for i in range(5):
        success = assign_population_fitness(population)
        if success and counter_fn():
            break
        population.speciate()
        data = population.to_dict()
        print_progress(data)
        mutator(population)

    score = run_env(data['best_genome'], env_name='CartPole-v0', render=True)
    print(f'best_fitness: {score}')
    return True


def print_progress(data):
    data_string = ''
    for val in ['generation', 'best_fitness', 'worst_fitness', 'mean_fitness']:
        data_string += f' {val}: {data[val]}'
    print(data_string)


if __name__ == "__main__":
    neat_cart_pole()
