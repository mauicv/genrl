"""RES Population Class."""

from src.debug.class_debug_decorator import add_inst_validator
from src.debug.population_validator import validate_population
from src.populations.population import Population


@add_inst_validator(env="TESTING", validator=validate_population)
class RESPopulation(Population):
    def __init__(
            self,
            genome_seeder,
            population_size=150):
        """Speciated Population of genomes used within the RES algorithm.

        Extends Population class.
        """
        # if genome_seeder
        super().__init__(population_size=population_size,
                         delta=None,
                         genome_seeder=genome_seeder,
                         metric=None)

    def rank_transform(self):
        """Orders targets by there fitness and then allocates each an
        evenly distributed fitness between -0.5, 0.5.
        """
        sorted_targets = sorted(self.genomes, key=lambda item: item.fitness)
        for index, target in enumerate(sorted_targets):
            target.fitness = index/len(sorted_targets) - 0.5
