"""Class that acts on genomes to mutate them.

Use:

if mutating a single genome:

    ```py
    m = Mutator(config)
    new_genome = m.mutate(genome)
    ```

if mutating an entire generation of genomes with speciation:

    ```
    m = Mutator(config)
    new_genomes = m.gen_step(genomes)
    ```

Config:
- c_1:
    pass
- c_2:
    pass
- c_3:
    pass
- delta:
    pass
- weight_mutation_likelyhood,
    pass
- weight_mutation_rate_random,
    pass
- weight_mutation_rate_uniform,
    pass
- gene_disable_rate,
    pass
- mutation_without_crossover_rate,
    pass
- insterspecies_mating_rate,
    pass
- new_node_probability,
    pass
- new_link_probability
    pass

"""

import numpy as np
from numpy.random import choice
from graph.genome import Genome


POPULATION                      = 150
C_1                             = 1.0
C_2                             = 1.0
C_3                             = 0.4
DELTA                           = 3.0
WEIGHT_MUTATION_LIKELYHOOD      = 0.8
WEIGHT_MUTATION_RATE_UNIFORM    = 0.1
WEIGHT_MUTATION_RATE_RANDOM     = 0.9
WEIGHT_MUTATION_VARIANCE        = 0.1
GENE_DISABLE_RATE               = 0.75
MUTATION_WITHOUT_CROSSOVER_RATE = 0.25
INSTERSPECIES_MATING_RATE       = 0.001
NEW_NODE_PROBABILITY            = 0.03
NEW_EDGE_PROBABILITY            = 0.05


class Mutator:
    def __init__(
            self,
            c_1=C_1,
            c_2=C_2,
            c_3=C_3,
            delta=DELTA,
            weight_mutation_likelyhood=WEIGHT_MUTATION_LIKELYHOOD,
            weight_mutation_rate_random=WEIGHT_MUTATION_RATE_RANDOM,
            weight_mutation_rate_uniform=WEIGHT_MUTATION_RATE_UNIFORM,
            weight_mutation_variance=WEIGHT_MUTATION_VARIANCE,
            gene_disable_rate=GENE_DISABLE_RATE,
            mutation_without_crossover_rate=MUTATION_WITHOUT_CROSSOVER_RATE,
            insterspecies_mating_rate=INSTERSPECIES_MATING_RATE,
            new_node_probability=NEW_NODE_PROBABILITY,
            new_edge_probability=NEW_EDGE_PROBABILITY):
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.delta = delta
        self.weight_mutation_likelyhood = weight_mutation_likelyhood
        self.weight_mutation_rate_random = weight_mutation_rate_random
        self.weight_mutation_rate_uniform = weight_mutation_rate_uniform
        self.weight_mutation_variance = weight_mutation_variance
        self.gene_disable_rate = gene_disable_rate
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.insterspecies_mating_rate = insterspecies_mating_rate
        self.new_node_probability = new_node_probability
        self.new_edge_probability = new_edge_probability

    def __call__(self, population):
        return self.gen_step(population)

    def mutate_weights(self, genome):
        new_genome = Genome.copy(genome)
        if np.random.uniform(0, 1, 1) < self.weight_mutation_likelyhood:
            edges = new_genome.edges
            random_nums = np.random.uniform(0, 1, len(edges))
            for edge, random_num in zip(edges, random_nums):
                if random_num < self.weight_mutation_rate_random:
                    perturbation = np.random \
                        .normal(0, self.weight_mutation_variance, 1)[0]
                    edge.weight += perturbation
                if random_num > self.weight_mutation_rate_random and \
                        random_num < self.weight_mutation_rate_random + \
                        self.weight_mutation_rate_uniform:
                    perturbation = np.random.uniform(
                        new_genome.weight_low,
                        new_genome.weight_high)
                    edge.weight = perturbation
        return new_genome

    def mutate_topology(self, genome):
        if np.random.uniform(0, 1, 1) < self.new_node_probability:
            print('here')
        if np.random.uniform(0, 1, 1) < self.new_edge_probability:
            print('here')

    def add_node(self, genome):
        edge = choice(genome.get_addmissable_edges())
        edge.disabled = True
        from_node, to_node = (edge.from_node, edge.to_node)
        layer_num = choice(range(from_node.layer_num + 1, to_node.layer_num))
        new_node = genome.add_node(layer_num)
        genome.add_edge(from_node, new_node)
        genome.add_edge(new_node, to_node)
        return genome

    def add_edge(self, genome):
        pass

    def mate(genome_1, genome_2):
        pass

    def gen_step(self, population):
        pass
