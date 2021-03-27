"""Factory methods for building specific genome architectures or copying genomes."""

from gerel.genome.genome import Genome
from gerel.genome.node import Node
from gerel.genome.edge import Edge


def from_genes(
        nodes_genes,
        edges,
        input_size=2,
        output_size=2,
        weight_low=-2,
        weight_high=2,
        depth=3):
    """Reconstructs a Genome class from the return value
    to_reduced_repr called on an instance of a Genome class.

    :param nodes_genes: First member of genome.to_reduced_repr,
        list of node properties and indices tuples. Takes form:

        [(layer_num, layer_ind, innov, weight, type), ...]

        where layer number is the genome layer index, layer_ind
        is the index in the layer, innov is the innovation number.
        weight is the node bias and type is one of 'input',
        'hidden' or 'output'.
    :param edges: Second member of genome.to_reduced_repr,
        tuple of edge properties and indices. Takes form:

        [(
            from_node.to_reduced_repr,
            to_node.to_reduced_repr,
            weight,
            innov,
            active
        ), ...]

        where from_node.to_reduced_repr and
        to_node.to_reduced_repr take the same form as an
        individual node in node_genes. weight is the
        edge weight, innov it's innovation number and
        active is True or False.
    :param input_size: Number of input nodes
    :param output_size: Number of output nodes
    :param weight_low: Maximum weight on node and edges
    :param weight_high: Minimum weight on node and edges
    :param depth: Number of layers in network.
    :return: Constructed genome.
    """

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
    """ Builds a minimal genome with specified inputs and
    outputs, weight bounds, depth and one connected node in
    the first layer.

    :param input_size: Number of input nodes
    :param output_size: Number of output nodes
    :param weight_low: Maximum weight on node and edges
    :param weight_high: Minimum weight on node and edges
    :param depth: Number of layers in network.
    :return: Constructed genome.
    """

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
        layer_dims=(5, 5, 5)):
    """ Builds a densely connected genome with specified
    inputs and outputs, weight bounds and depth with every
    node in one layer connected to every node in the
    subsequent layer.


    :param input_size: Number of input nodes
    :param output_size: Number of output nodes
    :param weight_low: Maximum weight on node and edges
    :param weight_high: Minimum weight on node and edges
    :param layer_dims: Tuple of layer dimensions
    :return: Constructed genome.
    """

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


def copy(genome):
    """Deep copy of genome instance.

    :param genome: Instance of Genome class
    :return: Copied instance of Genome glass
    """
    new_genome = Genome(
        input_size=len(genome.inputs),
        output_size=len(genome.outputs),
        weight_low=genome.weight_low,
        weight_high=genome.weight_high,
        depth=genome.depth)
    layers = [[Node.copy(node) for node in layer]
              for layer in genome.layers[1:-1]]
    new_genome.layers = [
        new_genome.inputs,
        *layers,
        new_genome.outputs
    ]
    nodes = [new_genome.layers[node.layer_num][node.layer_ind]
             for node in genome.nodes]
    new_genome.nodes = nodes
    for edge in genome.edges:
        new_edge = Edge.copy(edge, new_genome)
        new_genome.edge_innovs.add((
            new_edge.from_node.innov,
            new_edge.to_node.innov))
    return new_genome
