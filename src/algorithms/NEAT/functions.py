import numpy as np
from numpy.random import choice, uniform
from src.genome.genome import Genome


def curry_weight_mutator(
        weight_mutation_likelihood,
        weight_mutation_rate_random,
        weight_mutation_variance,
        weight_mutation_rate_uniform,
        weight_low=-2,
        weight_high=2,
        ):
    def weight_adj(target):
        if np.random.uniform(0, 1, 1) < weight_mutation_likelihood:
            if np.random.uniform(0, 1, 1) < weight_mutation_rate_random:
                perturbation = np.random \
                    .normal(0, weight_mutation_variance, 1)[0]
                target.weight += perturbation
            if weight_mutation_rate_random < np.random.uniform(0, 1) < \
                    weight_mutation_rate_random + weight_mutation_rate_uniform:
                perturbation = np.random.uniform(
                    weight_low,
                    weight_high)
                target.weight = perturbation
    return weight_adj


def pair_genes(primary, secondary):
    """For any list of genes with innov values, so edges or nodes we
    iterate through and chose one randomly when there exists a
    corresponding innov number and select the primary (fittest) gene when
    a match doesn't exist.
    """

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


def add_edge(genome):
    """When a edge is added we sample two layers without replacement and
    then order them. We then sample a node from each and add a new edge
    between them.
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


def add_node(genome):
    """When a node is added we randomly sample an admissible edge and then
    randomly sample an layer index in the range of layers the edge spans.
    After disabling the sampled edge we add a new node in the selected
    layer and then connect it with two new edges.
    """

    edge = choice(genome.get_addmissable_edges())
    from_node, to_node = (edge.from_node, edge.to_node)
    layer_num = choice(range(from_node.layer_num + 1, to_node.layer_num))
    new_node = genome.add_node(layer_num)
    genome.add_edge(from_node, new_node)
    genome.add_edge(new_node, to_node)
    edge.active = False
    return genome


def curry_crossover(gene_disable_rate):
    def crossover(primary=None, secondary=None):
        """Produces a child of the primary and secondary genomes.

        Note that genome.nodes excludes input and output layers.
        """
        node_genes = pair_genes(
            primary.nodes,
            secondary.nodes)
        edge_genes = pair_genes(
            primary.edges,
            secondary.edges)

        def activate_disabled(edge):
            (a, b, c, d, active) = edge
            if active:
                return edge
            if uniform(0, 1) > gene_disable_rate:
                edge = (a, b, c, d, True)
            return edge

        edge_genes = [activate_disabled(edge) for edge in edge_genes]
        new_genome = Genome.from_genes(
            node_genes,
            edge_genes,
            input_size=len(primary.inputs),
            output_size=len(primary.outputs),
            weight_low=primary.weight_low,
            weight_high=primary.weight_high,
            depth=primary.depth)
        return new_genome
    return crossover
