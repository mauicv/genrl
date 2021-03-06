"""Population Class.

The distance measure δ allows us to speciate using a compatibility threshold
δt. An ordered list of species is maintained. In each generation, genomes are
sequentially placed into species.

Compatibility Distance:

δ = c_1*E/N + c_2*D/N + c3 · W.

where E is excess and D is disjoint genes and W is the average weight
differences of matching genes including disabled genes
"""

from src.genome import Genome
from src.metrics import generate_neat_metric
from src.mutator import Mutator
from random import choice, random
from src.debug.class_debug_decorator import add_inst_validator
from src.debug.population_validator import validate_population

POPULATION                      = 150
DELTA                           = 3.0
MUTATION_WITHOUT_CROSSOVER_RATE = 0.25
INTERSPECIES_MATING_RATE        = 0.001
SPECIES_MEMBER_SURVIVAL_RATE    = 0.2


@add_inst_validator(env="TESTING", validator=validate_population)
class Population:
    def __init__(
            self,
            population_size=POPULATION,
            delta=DELTA,
            mutation_without_crossover_rate=MUTATION_WITHOUT_CROSSOVER_RATE,
            interspecies_mating_rate=INTERSPECIES_MATING_RATE,
            species_member_survival_rate=SPECIES_MEMBER_SURVIVAL_RATE,
            seed_genomes=None,
            mutator=Mutator()):
        self.population_size = population_size
        self.delta = delta
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.interspecies_mating_rate = interspecies_mating_rate
        self.species_member_survival_rate = species_member_survival_rate
        self.species = {}
        self.genomes = []

        self.mutator = mutator
        if not seed_genomes:
            seed_genomes = [Genome.default()]

        N = int(self.population_size / len(seed_genomes))

        for seed_genome in seed_genomes:
            for i in range(N):
                genome = Genome.copy(seed_genome)
                if i == 0:
                    self.genomes.append(genome)
                    continue
                self.mutator.mutate_weights(genome)
                self.mutator.mutate_topology(genome)
                self.genomes.append(genome)

    def step(self, metric=generate_neat_metric()):
        self.speciate(metric=metric)
        self.evolve()

    def evolve(self):
        """Takes previous population and each fitness and then evolves them into the next generation of genomes.

        - First we compute the population proportion that each group is granted.
        - Then we keep only the top species_member_survival_rate of each generation.
        - for each group
            - we put the top performing genome into the new populations
            - randomly draw Genomes from the remaining top performing genomes and apply mutations/pairing until the
            rest of the groups population share is taken up.

        """
        total_group_fitness_sum = sum([item['group_fitness'] for key, item in self.species.items()])
        new_genomes = []
        for key, item in self.species.items():
            pop_prop = int(round(self.population_size * (item['group_fitness']/total_group_fitness_sum)))
            survival_prop = int(len(item['group']) * self.species_member_survival_rate)
            survival_prop = 5 if survival_prop < 5 else survival_prop
            item['group'] = item['group'][:survival_prop]
            best_performer = Genome.copy(item['group'][0])
            new_genomes.append(best_performer)
            for _ in range(pop_prop - 1):
                selected_gene = choice(item['group'])
                new_genome = Genome.copy(selected_gene)
                new_genome.fitness = selected_gene.fitness
                self.mutator.mutate_weights(new_genome)
                self.mutator.mutate_topology(new_genome)
                if random() > self.mutation_without_crossover_rate:
                    other_genome = None
                    if random() < self.interspecies_mating_rate and len(self.species) > 1:
                        # select from other species
                        other_item = choice([item for _, item in self.species.items()])
                        if len(other_item['group']) > 2:
                            other_genome = choice([g for g in other_item['group'] if g is not selected_gene])
                    elif len(item['group']) > 2:
                        other_genome = choice([g for g in item['group'] if g is not selected_gene])
                    if other_genome:
                        secondary, primary = sorted([new_genome, other_genome], key=lambda g: g.fitness)
                        new_genome = self.mutator.mate(primary, secondary)
                new_genomes.append(new_genome)
        self.genomes = new_genomes

    def speciate(self, metric=generate_neat_metric()):
        self.species[1] = {
            'repr': self.genomes[0],
            'group': [self.genomes[0]]
        }

        for genome in self.genomes[1:]:
            assigned_group = False
            for key, item in self.species.items():
                if metric(genome, item['repr']) < self.delta:
                    assigned_group = True
                    self.species[key]['group'].append(genome)
                    break
            if not assigned_group:
                self.species[len(self.species)+1] = {
                    'repr': genome,
                    'group': [genome]
                }

        for key, item in self.species.items():
            group_size = len(item['group'])
            adj_fitness = lambda x: x.fitness/group_size
            group_fitness = sum([adj_fitness(g) for g in item['group']])
            item['group_fitness'] = group_fitness
            item['group'].sort(key=adj_fitness, reverse=True)