"""Class that acts on a genome or pair of genomes to mutate them."""

import numpy as np
from numpy.random import choice, uniform
from src.genome import Genome
from src.util import print_genome


WEIGHT_MUTATION_LIKELIHOOD      = 0.8
WEIGHT_MUTATION_RATE_UNIFORM    = 0.1
WEIGHT_MUTATION_RATE_RANDOM     = 0.9
WEIGHT_MUTATION_VARIANCE        = 0.1
GENE_DISABLE_RATE               = 0.75
NEW_NODE_PROBABILITY            = 0.03
NEW_EDGE_PROBABILITY            = 0.05


class Mutator:
    def __init__(
            self,
            weight_mutation_likelihood=WEIGHT_MUTATION_LIKELIHOOD,
            weight_mutation_rate_random=WEIGHT_MUTATION_RATE_RANDOM,
            weight_mutation_rate_uniform=WEIGHT_MUTATION_RATE_UNIFORM,
            weight_mutation_variance=WEIGHT_MUTATION_VARIANCE,
            gene_disable_rate=GENE_DISABLE_RATE,
            new_node_probability=NEW_NODE_PROBABILITY,
            new_edge_probability=NEW_EDGE_PROBABILITY):

        self.weight_mutation_likelihood = weight_mutation_likelihood
        self.weight_mutation_rate_random = weight_mutation_rate_random
        self.weight_mutation_rate_uniform = weight_mutation_rate_uniform
        self.weight_mutation_variance = weight_mutation_variance
        self.gene_disable_rate = gene_disable_rate
        self.new_node_probability = new_node_probability
        self.new_edge_probability = new_edge_probability

    def mutate_weights(self, genome):
        if np.random.uniform(0, 1, 1) < self.weight_mutation_likelihood:
            edges = genome.edges
            random_nums = np.random.uniform(0, 1, len(edges))
            for edge, random_num in zip(edges, random_nums):
                if random_num < self.weight_mutation_rate_random:
                    perturbation = np.random \
                        .normal(0, self.weight_mutation_variance, 1)[0]
                    edge.weight += perturbation
                if self.weight_mutation_rate_random < random_num < self.weight_mutation_rate_random + \
                        self.weight_mutation_rate_uniform:
                    perturbation = np.random.uniform(
                        genome.weight_low,
                        genome.weight_high)
                    edge.weight = perturbation
        return genome

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
        """

        # TODO: sometimes get_addmissable_edges returns emptylist. What should the behavour be in this case.
        edge = choice(genome.get_addmissable_edges())
        from_node, to_node = (edge.from_node, edge.to_node)
        layer_num = choice(range(from_node.layer_num + 1, to_node.layer_num))
        new_node = genome.add_node(layer_num)
        genome.add_edge(from_node, new_node)
        genome.add_edge(new_node, to_node)
        edge.active = False
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
        genome.add_edge(from_node, to_node)
        return genome

    def mate(self, primary=None, secondary=None):
        """Produces a child of the primary and secondary genomes.

        Note that genome.nodes excludes input and output layers."""

        node_genes = self.pair_genes(
            primary.nodes,
            secondary.nodes)
        edge_genes = self.pair_genes(
            primary.edges,
            secondary.edges)
        for _,_,_,_,active in edge_genes:
            if not active:
                if uniform(0, 1) > self.gene_disable_rate:
                    active = True
        new_genome = Genome.from_genes(
            node_genes,
            edge_genes,
            input_size=len(primary.inputs),
            output_size=len(primary.outputs),
            weight_low=primary.weight_low,
            weight_high=primary.weight_high,
            depth=primary.depth)
        return new_genome

    def pair_genes(self, primary, secondary):
        """For any list of genes with innov values, so edges or nodes we
        iterate through and chose one randomly when there exists a
        corresponding innov number and select the primary (fittest) gene when
        a match doesn't exist."""

        i, j = (0, 0)
        selected_gene = []
        while True:
            if primary[i].innov == secondary[j].innov:
                gene = choice([primary[i], secondary[j]])
                selected_gene.append(gene.to_reduced_repr)
                i, j = (i + 1, j + 1)
            elif primary[i].innov < secondary[j].innov:
                selected_gene.append(primary[i].to_reduced_repr)
                i += 1
            elif primary[i].innov > secondary[j].innov:
                j += 1

            if i == len(primary):
                break

            if j == len(secondary):
                excess = [gene.to_reduced_repr for gene in primary[i:]]
                selected_gene = selected_gene + excess
                break
        return selected_gene
