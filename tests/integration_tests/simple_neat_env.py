"""Example Implementation of NEAT algorithm from
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf

Note: The we use restricted networks in the sense all edges point forwards.
"""


import sys
import os

DIR = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))  # noqa
sys.path.insert(0, DIR)  # noqa

from src import Genome
from src import Population
from src import Mutator
from src import Model
from src import generate_neat_metric
from src.util import print_genome, print_population
from tqdm import tqdm


def test_neat_xor():
    pop_size = 150
    m = Mutator(
        new_edge_probability=0.1,
        new_node_probability=0.05
    )
    g = Genome.default(
        input_size=2,
        output_size=1,
        depth=3
    )
    p = Population(
        population_size=pop_size,
        seed_genomes=[g],
        mutator=m,
        delta=3
    )
    d = generate_neat_metric()

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
            yield data[i%4]

    sln_count = 0
    for i in tqdm(range(1000)):
        for g in p.genomes:
            fitness = 0
            best_fitness = 0
            model = Model(g.to_reduced_repr)
            for inputs, output in generate_data(4):
                pred = model(inputs)[0]
                pred = pred > 0
                fitness += pred == output
                if fitness > best_fitness:
                    best_fitness = fitness
            g.fitness = fitness/4
        if best_fitness == 4:
            sln_count += 1

        if sln_count == 10:
            break

        p.step(metric=d)

    fitnesses = []
    best_genome = None
    best_fitness = 0
    for genome in p.genomes:
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
        fitnesses.append(fitness)
    print()
    print('best_fitness:', best_fitness)
    print_population(p)
    print_genome(best_genome)
    return True

