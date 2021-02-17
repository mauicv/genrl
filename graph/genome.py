"""Genome class

Taken from NEAT implementation and adapted for layered networks.
see http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
"""

from random import random
import itertools


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
        self.weight_range = self.weight_high - self.weight_high

        self.inputs = [Node(0) for i in range(input_size)]
        self.outputs = [Node(1+depth) for j in
                        range(input_size, input_size + output_size)]

        self.layers = [self.inputs,
                       [Node(1)], *[[] for i in range(depth-1)],
                       self.outputs]

        for n in self.inputs:
            Edge(n, self.layers[1][0], self.sample_weight)

        for n in self.outputs:
             Edge(self.layers[1][0], n, self.sample_weight)

    @property
    def sample_weight(self):
        return random() * self.weight_range - self.weight_low

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
    def nodes(self):
        return [node for layer in self.layers for node in layer]

    def add_node(self, layer_index):
        if layer_index == 0 or layer_index == len(self.layers)-1:
            raise ValueError('Cannot add node to input or output layer')
        new_node = Node(layer_index)
        self.layers[layer_index].append(new_node)
        return new_node

    def add_edge(self, from_node, to_node):
        if to_node.layer_num - from_node.layer_num < 1:
            raise ValueError('Cannot connect edge to lower or same layer')
        return Edge(from_node, to_node, self.sample_weight)

class Node:
    """Node."""

    id_iter = itertools.count()
    def __init__(self, layer_num):
        self.layer_num = layer_num
        self.innov = next(Edge.innov_iter)
        self.edges_out = []
        self.edges_in = []

class Edge:
    """Edge."""

    innov_iter = itertools.count()
    def __init__(self, from_node, to_node, weight):
        self.disabled = False
        self.innov = next(Edge.innov_iter)
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight

        from_node.edges_out.append(self)
        to_node.edges_in.append(self)
