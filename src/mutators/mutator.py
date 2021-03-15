"""Class that acts on a genome or pair of genomes to mutate them."""

from src.genome.genome import Genome
from src.populations.population import Population


class Mutator:
    def __init__(self):
        pass

    def __call__(self, target):
        if isinstance(target, Genome):
            self.call_on_genome(target)
        elif isinstance(target, Population):
            self.call_on_population(target)

    def call_on_population(self, population):
        err_msg = 'Mutator call should act on Population Objects. Did you forget to overwrite call_on_population'
        raise NotImplementedError(err_msg)

    def call_on_genome(self, genome):
        err_msg = 'Mutator call should act on Genome Objects. Did you forget to overwrite call_on_genome'
        raise NotImplementedError(err_msg)
