from gerel.genome.factories import copy
from gerel.genome.genome import Genome
from gerel.populations.population import Population
from gerel.mutators.mutator import Mutator
from numpy.random import normal
import numpy as np


class SIMPLEMutator(Mutator):
    def __init__(self, std_dev, survival_rate):
        """ SIMPLE-ES algorithm mutator.

        Orders the population by fitness and retains the top by
        specified survival_rate.

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
        """Mutate Population.

        sorts genomes by fitness, removes any past a certain cutoff.
        Randomly samples from the remaining genomes and mutates them
        until the population is compelte again.

        :param population: Population Object being mutated
        :return: None
        """
        genomes = sorted(population.genomes, key=lambda g: g.fitness, reverse=True)
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
        """Mutate Genome.

        perturbs genomes weights by a value sampled from a normal
        distribution.

        :param genome: Genome to mutate.
        :return: None
        """
        new_weights = np.random.normal(loc=genome.weights, scale=self.std_dev)
        genome.weights = new_weights
