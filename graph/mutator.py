"""Class that acts on genomes to mutate them."""

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
        self.edge_innovations = {}

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
            self.add_node(genome)
        if np.random.uniform(0, 1, 1) < self.new_edge_probability:
            self.add_edge(genome)
        return genome

    def add_node(self, genome):
        """When a node is added we randomly sample an admissible edge and then
        randomly sample an layer index in the range of layers the edge spans.
        After disabling the sampled edge we add a new node in the selected
        layer and then connect it with two new edges.

        NOTE: We also update the edge_innovation dictionary in order to
        prevent duplicate edge innovations which can occur in add_edge.
        """

        edge = choice(genome.get_addmissable_edges())
        edge.disabled = True
        from_node, to_node = (edge.from_node, edge.to_node)
        layer_num = choice(range(from_node.layer_num + 1, to_node.layer_num))
        new_node = genome.add_node(layer_num)
        edge_1 = genome.add_edge(from_node, new_node)
        key_1 = (edge_1.from_node.innov, edge_1.to_node.innov)
        self.edge_innovations[key_1] = edge_1.innov
        edge_2 = genome.add_edge(new_node, to_node)
        key_2 = (edge_2.from_node.innov, edge_2.to_node.innov)
        self.edge_innovations[key_2] = edge_2.innov
        return genome

    def add_edge(self, genome):
        """When a edge is added we sample two layers without replacement and
        then order them. We then sample a node from each and add a new edge
        between them.

        NOTE: we also maintain a dictionary of edge_innovations as there is a
        possibility we may generate the same innovation twice.
        """

        non_empty_layers = [layer_num for layer_num in
                            range(len(genome.layers)) if
                            len(genome.layers[layer_num]) > 0]
        from_layer, to_layer = sorted(choice(
            non_empty_layers, 2, replace=False))
        from_node = choice(genome.layers[from_layer])
        to_node = choice(genome.layers[to_layer])
        key = (from_node.innov, to_node.innov)
        innov = self.edge_innovations.get(key, None)
        edge = genome.add_edge(from_node, to_node, innov=innov)
        self.edge_innovations[key] = edge.innov
        return genome

    def mate(self, genome_1, genome_2):
        nodes = self.pair_genes(genome_1.nodes, genome_2.nodes)
        edges = self.pair_genes(genome_1.edges, genome_2.edges)
        Genome.fromGenes(nodes, edges)

    def pair_genes(self, primary, secondary):
        i, j = (0, 0)
        selected_nodes = []
        while True:
            if primary[i].innov == secondary[j].innov:
                gene = choice([primary[i], secondary[j]])
                selected_nodes.append(gene.__class__.Copy(gene))
                i, j = (i + 1, j + 1)
            elif primary[i].innov < secondary[j].innov:
                selected_nodes.append(primary[i].__class__.Copy(gene))
                i += 1
            elif primary[i].innov > secondary[j].innov:
                j += 1

            if i == len(primary):
                break

            if j == len(secondary):
                excess = [gene.__class__.Copy(gene) for gene in primary[i:]]
                selected_nodes = selected_nodes + excess
                break
