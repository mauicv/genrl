"""RES Population Class."""

from src.debug.class_debug_decorator import add_inst_validator
from src.debug.population_validator import validate_population
from src.populations.population import Population
from src.populations.genome_seeders import curry_genome_seeder
from src.RES.mutator import RESMutator


@add_inst_validator(env="TESTING", validator=validate_population)
class RESPopulation(Population):
    def __init__(
            self,
            population_size=150,
            genome_seeder=curry_genome_seeder(mutator=RESMutator)):
        """Speciated Population of genomes used within the RES algorithm.

        Extends Population class.
        """
        super().__init__(population_size=population_size,
                         delta=None,
                         genome_seeder=genome_seeder,
                         metric=None)
