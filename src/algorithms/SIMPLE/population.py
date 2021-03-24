"""RES Population Class."""

from src.debug.class_debug_decorator import add_inst_validator
from src.debug.population_validator import validate_population
from src.populations.population import Population


@add_inst_validator(env="TESTING", validator=validate_population)
class SIMPLEPopulation(Population):
    def __init__(
            self,
            genome_seeder,
            population_size=150):
        """Non-Speciated Population of genomes used within the SIMPLE algorithm.

        Extends Population class.
        """
        # if genome_seeder
        super().__init__(population_size=population_size,
                         delta=None,
                         genome_seeder=genome_seeder,
                         metric=None)