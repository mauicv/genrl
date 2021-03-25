"""Class that acts on a genome or pair of genomes to mutate them using the REINFORCE-ES algorithm."""

from src.genome.genome import Genome
from src.populations.population import Population
from src.mutators.mutator import Mutator
from numpy.random import normal
import numpy as np
from src.util.math import vector_decomp


class RESMutator(Mutator):
    def __init__(self, initial_mu, std_dev, alpha=0.0001):
        """ REINFORCE-ES algorithm mutator.

        :param initial_mu: The initial mean for the normal distribution from which
            we sample new genomes.
        :param std_dev: The standard deviation of the normal distribution.
        :param alpha: The size of update

        :raises ValueError: If initial_mu is not a non-empty list or mu and std_dev not the
            same size
        """
        super().__init__()
        if not isinstance(initial_mu, list) and not isinstance(initial_mu, np.ndarray):
            raise ValueError('initial_mu must be a list or numpy array')
        if len(initial_mu) == 0:
            raise ValueError('initial_mu must have length greater than 0')
        if isinstance(std_dev, list) and len(initial_mu) != len(std_dev):
            raise ValueError('std_dev must be a float value or a list of the same size as mu.')

        self.mu = np.array(initial_mu)
        self.std_dev = np.array(std_dev)
        self.alpha = alpha

    def __call__(self, target):
        if isinstance(target, Genome):
            self.call_on_genome(target)
        elif isinstance(target, Population):
            self.call_on_population(target)

    def compute_derivative(self, targets):
        grad_pi = np.zeros_like(self.mu, dtype='float64')
        for target in targets:
            f, z = target.values()
            grad_pi += f*(np.array(z) - self.mu)/(self.std_dev**2)
        return grad_pi*self.alpha/len(targets)

    def _apply_derivative(self, diff):
        self.mu = [d+m for d, m in zip(diff, self.mu)]

    def call_on_population(self, population, apply_rank_transform=True):
        if apply_rank_transform:
            population.rank_transform()
        derivative = self.compute_derivative(population.genomes)
        self._apply_derivative(derivative)
        for genome in population.genomes:
            self.call_on_genome(genome)
        population.generation += 1

    def call_on_genome(self, genome):
        new_weights = np.random.normal(loc=self.mu, scale=self.std_dev)
        genome.weights = new_weights


class ADRESMutator(RESMutator):
    def __init__(self, initial_mu, std_dev):
        super().__init__(initial_mu, std_dev)

    def compute_genome_preference(self, genome, derivative):
        fitness, weights = genome.values()
        weights, derivative = (np.array(weights), np.array(derivative))
        w_parallel, w_orthogonal = vector_decomp(weights, self.mu, derivative)
        if w_parallel > 0:
            preference = ((1 - w_orthogonal) ** 2) * fitness
            return preference if preference > 0 else 0
        else:
            return 0

    def update_mu(self, population, derivative):
        preference = lambda g: self.compute_genome_preference(g, derivative)
        best_genome = max(population.genomes, key=preference)
        _, new_mu = best_genome.values()
        self.mu = new_mu

    def call_on_population(self, population, apply_rank_transform=True):
        if apply_rank_transform:
            population.rank_transform()
        derivative = self.compute_derivative(population.genomes)
        normed_derivative = derivative/np.linalg.norm(derivative)
        self.update_mu(population, normed_derivative)
        for genome in population.genomes:
            self.call_on_genome(genome)
        population.generation += 1
