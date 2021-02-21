"""Genome class

Taken from NEAT implementation and adapted for layered networks.
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
"""
import random
from graph.edge import Edge
from graph.node import Node


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
        self.nodes = []
        self.edges = []

    @classmethod
    def from_genes(cls, nodes, edges):
        pass

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
            genome.edges.append(
                Edge(n, genome.layers[1][0], genome.sample_weight))
        for n in genome.outputs:
            genome.edges.append(
                Edge(genome.layers[1][0], n, genome.sample_weight))
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
        nodes = [new_genome.layers[node.layer_num][node.layer_ind]
                 for node in genome.nodes]
        new_genome.nodes = nodes
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
        layer and are not disabled."""

        addmissable = lambda e: e.to_node.layer_num - e.from_node.layer_num \
            > 1 and not e.disabled
        return [edge for layer in self.layers
                for node in layer
                for edge in node.edges_out
                if addmissable(edge)]

    def add_node(self, layer_num):
        if layer_num == 0 or layer_num == len(self.layers) - 1:
            raise ValueError('Cannot add node to input or output layer')
        new_node = Node(layer_num, len(self.layers[layer_num]))
        self.layers[layer_num].append(new_node)
        self.nodes.append(new_node)
        return new_node

    def add_edge(self, from_node, to_node, innov=None):
        edge = Edge(from_node, to_node, self.sample_weight, innov=innov)
        self.edges.append(edge)
        return edge
