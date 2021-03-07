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
from src import BatchJob
from src import generate_neat_metric
from random import choice
from src.util import print_genome

def test_neat_xor():
    pop_size = 30
    m = Mutator()
    g = Genome.default(
        input_size=2,
        output_size=1,
        depth=2
    )
    p = Population(
        population_size=pop_size,
        seed_genomes=[g],
        mutator=m,
        delta=1
    )
    d = generate_neat_metric()

    def xor(a, b):
        return bool(a) != bool(b)

    def generate_data(n):
        for _ in range(n):
            vals = [True, False]
            a = choice(vals)
            b = choice(vals)
            c = xor(a, b)
            yield [a, b], c

    # datagen = generate_data()
    for i in range(10):
        for g in p.genomes:
            print_genome(g)
            for inputs, output in generate_data(1):
                model = Model(g.to_reduced_repr)
                # print(model.cells)
                pred = model(inputs)

                print('inputs:', inputs, 'output:', output, 'pred:', pred)
        p.step(d)


    # print(population.genomes)
    return False

