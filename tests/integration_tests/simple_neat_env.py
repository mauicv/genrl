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


def neat_test():
    mutator = Mutator()
    seed_genome = Genome.default()
    population = Population(mutator=mutator, seed_genomes=[seed_genome])
    metric = generate_neat_metric()
    # test population

    # print(population.genomes)
    return False

