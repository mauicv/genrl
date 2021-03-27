"""Class that acts on a genome or population of genomes.

The call_on_population and call_on_genome methods are
expected to be overwritten.
"""

from gerel.genome.genome import Genome
from gerel.populations.population import Population


class Mutator:
    def __init__(self):
        pass

    def __call__(self, target):
        """Mutates a target genome or population in place.

        :param target: Population or Genome.
        :return: None
        :raises ValueError: If target not a Genome or Population.
        """
        if isinstance(target, Genome):
            self.call_on_genome(target)
        elif isinstance(target, Population):
            self.call_on_population(target)
        else:
            raise ValueError('Target should be either a Genome or a Population')

    def call_on_population(self, population):
        err_msg = 'Mutator call should act on Population Objects. Did you forget to overwrite call_on_population'
        raise NotImplementedError(err_msg)

    def call_on_genome(self, genome):
        err_msg = 'Mutator call should act on Genome Objects. Did you forget to overwrite call_on_genome'
        raise NotImplementedError(err_msg)
