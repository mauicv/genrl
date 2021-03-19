from src.genome.genome import Genome
from src.genome.node import Node
from src.genome.edge import Edge


def from_genes(
        nodes_genes,
        edges,
        input_size=2,
        output_size=2,
        weight_low=-2,
        weight_high=2,
        depth=3):

    new_genome = Genome(
        input_size=input_size,
        output_size=output_size,
        weight_low=weight_low,
        weight_high=weight_high,
        depth=depth)

    layer_maxes = [0 for _ in range(depth)]
    for node_gene in nodes_genes:
        layer_num, layer_ind, _, weight, _ = node_gene
        if layer_maxes[layer_num - 1] < layer_ind + 1:
            layer_maxes[layer_num - 1] = layer_ind + 1

    layers = []
    for layer_max in layer_maxes:
        layers.append([None for _ in range(layer_max)])

    nodes = []
    for node_gene in nodes_genes:
        layer_num, layer_ind, innov, weight, node_type = node_gene
        node = Node(layer_num, layer_ind, weight, type=node_type)
        nodes.append(node)
        layers[layer_num - 1][layer_ind] = node

    new_genome.layers = [new_genome.inputs, *layers, new_genome.outputs]

    new_genome.nodes = nodes
    for from_node_reduced, to_node_reduced, weight, innov, active in edges:
        from_layer_num, from_layer_ind, _, _, _ = from_node_reduced
        from_node = new_genome.layers[from_layer_num][from_layer_ind]
        to_layer_num, to_layer_ind, _, _, _ = to_node_reduced
        to_node = new_genome.layers[to_layer_num][to_layer_ind]
        edge = Edge(from_node, to_node, weight, active)
        new_genome.edges.append(edge)
        new_genome.edge_innovs.add((from_node.innov, to_node.innov))
    return new_genome


def minimal(
        input_size=2,
        output_size=2,
        weight_low=-2,
        weight_high=2,
        depth=3):

    genome = Genome(
        input_size=input_size,
        output_size=output_size,
        weight_low=weight_low,
        weight_high=weight_high,
        depth=depth)
    genome.layers = [genome.inputs,
                     *[[] for _ in range(depth)],
                     genome.outputs]
    genome.add_node(1)
    for n in genome.inputs:
        genome.add_edge(n, genome.layers[1][0])
    for n in genome.outputs:
        genome.add_edge(genome.layers[1][0], n)
    return genome


def dense(
        input_size=2,
        output_size=2,
        weight_low=-2,
        weight_high=2,
        layer_dims=(5, 5, 5)
        ):

    genome = Genome(
        input_size=input_size,
        output_size=output_size,
        weight_low=weight_low,
        weight_high=weight_high,
        depth=len(layer_dims))

    genome.layers = [genome.inputs,
                     *[[] for _ in range(len(layer_dims))],
                     genome.outputs]

    layer_ind = 1
    for layer_size in layer_dims:
        for _ in range(layer_size):
            genome.add_node(layer_ind)
        layer_ind += 1

    for layer_1, layer_2 in zip(genome.layers, genome.layers[1:]):
        for node_1 in layer_1:
            for node_2 in layer_2:
                genome.add_edge(node_1, node_2)

    return genome
