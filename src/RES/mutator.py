"""Class that acts on a genome or pair of genomes to mutate them using the REINFORCE-ES algorithm."""

from src.genome.genome import Genome
from src.populations.population import Population
from src.mutators.mutator import Mutator


class RESMutator(Mutator):
    def __init__(self, initial_mu):
        super().__init__()
        self.mu = initial_mu

    def __call__(self, target):
        if isinstance(target, Genome):
            self.call_on_genome(target)
        elif isinstance(target, Population):
            self.call_on_population(target)

    def call_on_population(self, population):
        pass

    def call_on_genome(self, genome):
        pass