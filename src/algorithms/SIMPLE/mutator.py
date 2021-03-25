"""SIMPLE-ES Algorithm

orders the population by fitness and retains the top survival_rate
"""

from src.genome.factories import copy
from src.genome.genome import Genome
from src.populations.population import Population
from src.mutators.mutator import Mutator
from numpy.random import normal
import numpy as np


class SIMPLEMutator(Mutator):
    def __init__(self, std_dev, survival_rate):
        """ SIMPLE-ES algorithm mutator.

        :param std_dev: The standard deviation of the normal distribution.
        :param survival_rate: The proportion of population members that survive.
        """
        super().__init__()
        self.std_dev = np.array(std_dev)
        self.survival_rate = survival_rate

    def __call__(self, target):
        if isinstance(target, Genome):
            self.call_on_genome(target)
        elif isinstance(target, Population):
            self.call_on_population(target)

    def call_on_population(self, population):
        genomes = sorted(population.genomes, key=lambda g: g.fitness, reverse=True)
        # print([g.fitness for g in population.genomes])
        cutoff = int(self.survival_rate * population.population_size)
        genomes = genomes[0: cutoff]
        population.genomes = [*genomes]
        new_genomes = np.random.choice(genomes, population.population_size - cutoff)
        for genome in new_genomes:
            new_genome = copy(genome)
            self.call_on_genome(new_genome)
            population.genomes.append(new_genome)
        population.generation += 1

    def call_on_genome(self, genome):
        new_weights = np.random.normal(loc=genome.weights, scale=self.std_dev)
        genome.weights = new_weights
