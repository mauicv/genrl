from gerel.algorithms.NEAT.metric import generate_neat_metric
from gerel.debug.class_debug_decorator import add_inst_validator
from gerel.debug.population_validator import validate_population
from gerel.populations.population import Population
from gerel.populations.genome_seeders import curry_genome_seeder
from gerel.algorithms.NEAT.mutator import NEATMutator


@add_inst_validator(env="TESTING", validator=validate_population)
class NEATPopulation(Population):
    def __init__(
            self,
            population_size=150,
            delta=3.0,
            genome_seeder=curry_genome_seeder(NEATMutator()),
            metric=generate_neat_metric(1, 1, 3)):
        """Speciated Population of genomes used within the NEAT algorithm. Extends Population class.

        Notes:
          - The distance metric allows us to speciate using a compatibility threshold
          delta. An ordered list of species is maintained. In each generation, genomes are
          sequentially placed into species.

        :param genome_seeder: An iterable of genomes that will be used to generate the
            initial population.
        :param population_size: An integer number giving the size of the total Population.
        :param delta: A float number that defines the minimum distance two genomes require
            to be apart before there sorted into different species. Set to None if
            single species is desired (no speciation).
        :param metric: Function that takes two genomes and returns the distance between them.
            The metric is used to partition the genomes into species. If None then no
            speciation will take place.
        """
        super().__init__(population_size=population_size,
                         delta=delta,
                         genome_seeder=genome_seeder,
                         metric=metric)
