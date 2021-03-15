"""NEAT Population Class.

The distance measure δ allows us to speciate using a compatibility threshold
δt. An ordered list of species is maintained. In each generation, genomes are
sequentially placed into species.

Compatibility Distance:

δ = c_1*E/N + c_2*D/N + c3 · W.

where E is excess and D is disjoint genes and W is the average weight
differences of matching genes including disabled genes
"""

from src.NEAT.metric import generate_neat_metric
from src.debug.class_debug_decorator import add_inst_validator
from src.debug.population_validator import validate_population
from src.populations.population import Population
from src.populations.genome_seeders import default_genome_seeder


@add_inst_validator(env="TESTING", validator=validate_population)
class NEATPopulation(Population):
    def __init__(
            self,
            population_size=150,
            delta=3.0,
            genome_seeder=default_genome_seeder,
            metric=generate_neat_metric(1, 1, 3)):
        """Speciated Population of genomes used within the NEAT algorithm. Extends Population class.

        Notes:
          - The distance metric allows us to speciate using a compatibility threshold
          delta. An ordered list of species is maintained. In each generation, genomes are
          sequentially placed into species.
        """
        super().__init__(population_size=population_size,
                         delta=delta,
                         genome_seeder=genome_seeder,
                         metric=metric)
