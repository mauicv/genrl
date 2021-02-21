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
    def from_genes(
            cls,
            nodes_genes,
            edges,
            input_size=2,
            output_size=2,
            weight_low=-2,
            weight_high=2,
            depth=3):

        new_genome = cls(
            input_size=input_size,
            output_size=output_size,
            weight_low=weight_low,
            weight_high=weight_high,
            depth=depth)

        layer_maxes = [0 for i in range(depth)]
        for node_gene in nodes_genes:
            layer_num, layer_ind, _ = node_gene
            if layer_maxes[layer_num - 1] < layer_ind + 1:
                layer_maxes[layer_num - 1] = layer_ind + 1

        layers = []
        for layer_max in layer_maxes:
            layers.append([None for _ in range(layer_max)])

        nodes = []
        for node_gene in nodes_genes:
            layer_num, layer_ind, innov = node_gene
            node = Node(layer_num, layer_ind, innov=innov)
            nodes.append(node)
            layers[layer_num - 1][layer_ind] = node

        new_genome.layers = [new_genome.inputs, *layers, new_genome.outputs]

        new_genome.nodes = nodes
        for from_node_reduced, to_node_reduced, weight, innov in edges:
            from_layer_num, from_layer_ind, _ = from_node_reduced
            from_node = new_genome.layers[from_layer_num][from_layer_ind]
            to_layer_num, to_layer_ind, _ = to_node_reduced
            to_node = new_genome.layers[to_layer_num][to_layer_ind]
            edge = Edge(from_node, to_node, weight, innov=innov)
            new_genome.edges.append(edge)
        return new_genome

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

    def __repr__(self):
        repr_str = '\nNodes: \n'
        for node in self.nodes:
            repr_str += '\t' + str(node.to_reduced_repr) + '\n'
        repr_str += 'Edges: \n'
        for edge in self.edges:
            repr_str += '\t' + str(edge.to_reduced_repr) + '\n'
        # repr_str = '\nNodes: \n'
        # for node in self.nodes[0:5]:
        #     repr_str += '\t' + str(node.to_reduced_repr) + '\n'
        # if len(self.nodes) > 5:
        #     repr_str += '\t.\n\t.\n\t.\n'
        #     repr_str += '\t' + str(self.nodes[-1].to_reduced_repr) + '\n'
        # repr_str += 'Edges: \n'
        # for edge in self.edges[0:5]:
        #     repr_str += '\t' + str(edge.to_reduced_repr) + '\n'
        # if len(self.edges) > 5:
        #     repr_str += '\t.\n\t.\n\t.\n'
        #     repr_str += '\t' + str(self.edges[-1].to_reduced_repr) + '\n'
        return repr_str

    # def save(self):
    #     pass
    #
    # def load(self):
    #     pass
