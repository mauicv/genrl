from gerel.debug.class_debug_decorator import add_inst_validator
from gerel.debug.population_validator import validate_population
from gerel.populations.population import Population


@add_inst_validator(env="TESTING", validator=validate_population)
class SIMPLEPopulation(Population):
    def __init__(
            self,
            genome_seeder,
            population_size=150):
        """Non-Speciated Population of genomes used within the SIMPLE algorithm.

        Extends Population class.

        :param genome_seeder: An iterable of genomes that will
            be used to generate the initial population.
        :param population_size: An integer number giving the size
            of the total Population.
        """
        super().__init__(population_size=population_size,
                         delta=None,
                         genome_seeder=genome_seeder,
                         metric=None)
