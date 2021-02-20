"""Genome class

Taken from NEAT implementation and adapted for layered networks.
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
"""
import random
import itertools


def get_random():
    return random.random() * 2 - 1


class Genome:
    """Genome."""
    def __init__(
            self,
            input_size=2,
            output_size=2,
            weight_low=-2,
            weight_high=2,
            depth=3):

        self.depth = depth
        self.weight_low = weight_low
        self.weight_high = weight_high
        self.weight_range = self.weight_high - self.weight_low
        self.inputs = [Node(0, i, innov=-1) for i in range(input_size)]
        self.outputs = [Node(1 + depth, j, innov=-1) for j in
                        range(output_size)]
        self.layers = None

    @classmethod
    def default(
            cls,
            input_size=2,
            output_size=2,
            weight_low=-2,
            weight_high=2,
            depth=3):
        genome = cls(
            input_size=input_size,
            output_size=output_size,
            weight_low=weight_low,
            weight_high=weight_high,
            depth=depth)

        genome.layers = [genome.inputs,
                         *[[] for i in range(depth)],
                         genome.outputs]
        genome.add_node(1)

        for n in genome.inputs:
            Edge(n, genome.layers[1][0], genome.sample_weight)

        for n in genome.outputs:
            Edge(genome.layers[1][0], n, genome.sample_weight)
        return genome

    @classmethod
    def copy(cls, genome):
        new_genome = cls(
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

        for edge in genome.edges:
            Edge.copy(edge, new_genome)
        return new_genome

    @property
    def sample_weight(self):
        return random.random() * self.weight_range + self.weight_low

    def layer_edges_out(self, layer_num):
        return [edge for node in self.layers[layer_num]
                for edge in node.edges_out]

    def layer_edges_in(self, layer_num):
        return [edge for node in self.layers[layer_num]
                for edge in node.edges_in]

    def get_addmissable_edges(self):
        """Addmissable edges are defined as those that span more than one
        layer."""

        addmissable = lambda e: e.to_node.layer_num - e.from_node.layer_num > 1
        return [edge for layer in self.layers
                for node in layer
                for edge in node.edges_out
                if addmissable(edge)]

    @property
    def edges(self):
        """Returns all available edges."""

        return [edge for layer in self.layers for node in layer
                for edge in node.edges_out]

    @property
    def nodes(self):
        return [node for layer in self.layers for node in layer]

    def add_node(self, layer_num):
        if layer_num == 0 or layer_num == len(self.layers) - 1:
            raise ValueError('Cannot add node to input or output layer')
        new_node = Node(layer_num, len(self.layers[layer_num]))
        self.layers[layer_num].append(new_node)
        return new_node

    def add_edge(self, from_node, to_node):
        return Edge(from_node, to_node, self.sample_weight)


class Node:
    """Node."""

    innov_iter = itertools.count()

    def __init__(self, layer_num, layer_ind, innov=None):
        self.layer_num = layer_num
        self.layer_ind = layer_ind
        self.innov = innov if innov is not None else next(Node.innov_iter)
        self.edges_out = []
        self.edges_in = []

    @classmethod
    def copy(cls, node):
        """Copies the node location and innovation number.

        Note the edges_out and edges_in are not copied. This is taken care of
        in the Edge initialization step itself.
        """
        return cls(
            node.layer_num,
            node.layer_ind,
            innov=node.innov)


class Edge:
    """Edge."""

    innov_iter = itertools.count()

    def __init__(self, from_node, to_node, weight, innov=None):
        if to_node.layer_num - from_node.layer_num < 1:
            raise ValueError('Cannot connect edge to lower or same layer')
        self.disabled = False
        self.innov = innov if innov is not None else next(Edge.innov_iter)
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        from_node.edges_out.append(self)
        to_node.edges_in.append(self)

    @classmethod
    def copy(cls,
             edge,
             new_genome):
        fn = edge.from_node
        tn = edge.to_node

        if len(new_genome.layers) - 1 < fn.layer_num or \
                len(new_genome.layers[fn.layer_num]) - 1 < fn.layer_ind:
            raise ValueError('from_node does not exist on new_genome.')

        if len(new_genome.layers) - 1 < tn.layer_ind or \
                len(new_genome.layers[tn.layer_num]) - 1 < tn.layer_ind:
            raise ValueError('to_node does not exist on new_genome.')

        from_node = new_genome.layers[fn.layer_num][fn.layer_ind]
        to_node = new_genome.layers[tn.layer_num][tn.layer_ind]
        return cls(from_node, to_node, edge.weight, innov=edge.innov)
