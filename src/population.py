"""Population Class.

The distance measure δ allows us to speciate using a compatibility threshold
δt. An ordered list of species is maintained. In each generation, genomes are
sequentially placed into species.

# TODO:
    - compatibility distance
    - speciation
    - explicit fitness sharing


Compatibility Distance:

δ = c_1*E/N + c_2*D/N + c3 · W.

where E is excess and D is disjoint genes and W is the average weight
differences of matching genes including disabled genes
"""

from src.genome import Genome
from src.metrics import generate_neat_metric

POPULATION                      = 150
DELTA                           = 3.0
MUTATION_WITHOUT_CROSSOVER_RATE = 0.25
INTERSPECIES_MATING_RATE        = 0.001


class Population:
    def __init__(
            self,
            mutator,
            population_size=POPULATION,
            delta=DELTA,
            mutation_without_crossover_rate=MUTATION_WITHOUT_CROSSOVER_RATE,
            interspecies_mating_rate=INTERSPECIES_MATING_RATE):
        self.population_size = population_size
        self.delta = delta
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.interspecies_mating_rate = interspecies_mating_rate
        self.centers = []
        self.species = {}
        self.genomes = []
        self.mutator = mutator

    def populate(self, seed_genome=Genome.default()):
        for i in range(self.population_size):
            genome = Genome.copy(seed_genome)
            self.mutator.mutate_weights(genome)
            self.genomes.append(genome)

    def sort(self, metric=generate_neat_metric()):
        pass
