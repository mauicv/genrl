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
from src import curry_genome_seeder
from src import Model
from src import generate_neat_metric
from src.util.util import print_genome, print_population
from tqdm import tqdm


def neat_xor_example():
    pop_size = 150
    mutator = NEATMutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    seed_genome = minimal(
        input_size=2,
        output_size=1,
        depth=3
    )
    seeder = curry_genome_seeder(
        mutator=mutator,
        seed_genomes=[seed_genome]
    )
    metric = generate_neat_metric()
    population = NEATPopulation(
        population_size=pop_size,
        delta=3,
        genome_seeder=seeder,
        metric=metric
    )

    def xor(a, b):
        return bool(a) != bool(b)

    def generate_data(n):
        data = [
            [[True, True], xor(True, True)],
            [[True, False], xor(True, False)],
            [[False, True], xor(False, True)],
            [[False, False], xor(False, False)]
        ]
        for i in range(n):
            yield data[i % 4]

    sln_count = 0
    best_fitness = None
    for _ in tqdm(range(1000)):
        for genome in population.genomes:
            fitness = 0
            best_fitness = 0
            model = Model(genome.to_reduced_repr)
            for inputs, output in generate_data(4):
                pred = model(inputs)[0]
                pred = pred > 0
                fitness += pred == output
                if fitness > best_fitness:
                    best_fitness = fitness
            genome.fitness = fitness/4
        if best_fitness == 4:
            sln_count += 1

        if sln_count == 10:
            break

        population.speciate()
        mutator(population)

    fitness_scores = []
    best_genome = None
    best_fitness = 0
    for genome in population.genomes:
        fitness = 0
        model = Model(genome.to_reduced_repr)
        for inputs, output in generate_data(4):
            pred = model(inputs)[0]
            pred = pred > 0
            fitness += pred == output
        fitness = fitness / 4
        if fitness > best_fitness:
            best_fitness = fitness
            best_genome = genome
        fitness_scores.append(fitness)
    print()
    print('best_fitness:', best_fitness)
    print_population(population)
    print_genome(best_genome)
    return True


if __name__ == "__main__":
    neat_xor_example()
