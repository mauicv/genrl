import numpy as np
from numpy.random import choice
from gerel.genome.factories import copy
from random import random
from gerel.mutators.mutator import Mutator
from gerel.algorithms.NEAT.functions import curry_weight_mutator, curry_crossover, add_node, add_edge


class NEATMutator(Mutator):
    def __init__(
            self,
            weight_mutation_likelihood=0.8,
            weight_mutation_rate_random=0.1,
            weight_mutation_rate_uniform=0.9,
            weight_mutation_variance=0.1,
            mutation_without_crossover_rate=0.25,
            interspecies_mating_rate=0.001,
            species_member_survival_rate=0.2,
            gene_disable_rate=0.75,
            new_node_probability=0.03,
            new_edge_probability=0.05,
            weight_low=-2,
            weight_high=2,
            ):
        """build NEATMutator object that acts on NEATPopulation objects.

        :param weight_mutation_likelihood: Given a node or edge this
            is the likelihood that targets weight is mutated.
        :param weight_mutation_rate_random: Likelihood of normal
            distribution perturbation of weight.
        :param weight_mutation_variance: Variance of normal distribution
            used to perturb weights.
        :param mutation_without_crossover_rate: Likelihood of just
            weight or topological mutation occurring without crossover.
        :param interspecies_mating_rate: Likelihood of crossover between
            two species.
        :param species_member_survival_rate: Proportion of species
            members that survive yeah generation.
        :param gene_disable_rate: Likelihood of disabled gene staying
            inactive.
        :param new_node_probability: Likelihood of new node mutation.
        :param new_edge_probability: Likelihood of new edge mutation.
        :param weight_low: uniform distribution lower bound
        :param weight_high: uniform distribution upper bound
        """
        super().__init__()
        self.gene_disable_rate = gene_disable_rate
        self.new_node_probability = new_node_probability
        self.new_edge_probability = new_edge_probability
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.interspecies_mating_rate = interspecies_mating_rate
        self.species_member_survival_rate = species_member_survival_rate

        self.weight_mutator = curry_weight_mutator(
            weight_mutation_likelihood,
            weight_mutation_rate_random,
            weight_mutation_variance,
            weight_mutation_rate_uniform,
            weight_low=weight_low,
            weight_high=weight_high,
        )

        self.crossover = curry_crossover(gene_disable_rate)

    def call_on_population(self, population):
        """Takes population of speciated genomes and evolves them into the next generation of genomes.

        - First we compute the population proportion that each group is granted.
        - Then we keep only the top species_member_survival_rate of each generation.
        - for each group
            - we put the top performing genome into the new populations
            - randomly draw Genomes from the remaining top performing
                genomes and apply mutations/pairing until the rest of the
                groups population share is taken up.

        :param population: NEATPopulation object
        :return: None
        """
        total_group_fitness_sum = sum([item['group_fitness'] for key, item in population.species.items()])
        new_genomes = []
        for key, item in population.species.items():
            pop_prop = int(round(population.population_size * (item['group_fitness'] / total_group_fitness_sum)))
            survival_prop = int(len(item['group']) * self.species_member_survival_rate)
            survival_prop = 5 if survival_prop < 5 else survival_prop
            item['group'] = item['group'][:survival_prop]
            best_performer = copy(item['group'][0])
            new_genomes.append(best_performer)
            for _ in range(pop_prop - 1):
                selected_gene = choice(item['group'])
                new_genome = self.call_on_genome(selected_gene)
                if random() > self.mutation_without_crossover_rate:
                    other_genome = None
                    if random() < self.interspecies_mating_rate and len(population.species) > 1:
                        # select from other species
                        other_item = choice([item for _, item in population.species.items()])
                        if len(other_item['group']) > 2:
                            other_genome = choice([g for g in other_item['group'] if g is not selected_gene])
                    elif len(item['group']) > 2:
                        other_genome = choice([g for g in item['group'] if g is not selected_gene])
                    if other_genome:
                        secondary, primary = sorted([new_genome, other_genome], key=lambda g: g.fitness)
                        new_genome = self.crossover(primary, secondary)
                new_genomes.append(new_genome)
        population.generation += 1
        population.genomes = new_genomes

    def call_on_genome(self, genome):
        """Action on genome.

        Only performs weight and topological mutations.

        :param genome: Genome to copy and mutate.
        :return: New genome.
        """
        new_genome = copy(genome)
        new_genome.fitness = genome.fitness

        for edge in new_genome.edges:
            self.weight_mutator(edge)
        for node in new_genome.nodes:
            self.weight_mutator(node)

        if np.random.uniform(0, 1, 1) < self.new_node_probability:
            add_node(new_genome)

        if np.random.uniform(0, 1, 1) < self.new_edge_probability:
            add_edge(new_genome)

        return new_genome
