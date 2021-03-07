"""Genome class."""
from src.edge import Edge
from src.node import Node
from src.util import sample_weight
from src.debug.class_debug_decorator import add_inst_validator
from src.debug.genome_validator import validate_genome


@add_inst_validator(env="TESTING", validator=validate_genome)
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
        self.edge_innovs = set()

        self.inputs = [Node(0, i, 0, type='input')
                       for i in range(input_size)]
        self.outputs = [Node(1 + depth, j, 0, type='output')
                        for j in range(output_size)]
        self.layers = None
        self.nodes = []
        self.edges = []
        self.fitness = None

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
            layer_num, layer_ind, _, weight, _ = node_gene
            if layer_maxes[layer_num - 1] < layer_ind + 1:
                layer_maxes[layer_num - 1] = layer_ind + 1

        layers = []
        for layer_max in layer_maxes:
            layers.append([None for _ in range(layer_max)])

        nodes = []
        for node_gene in nodes_genes:
            layer_num, layer_ind, innov, weight, type = node_gene
            node = Node(layer_num, layer_ind, weight, type=type)
            nodes.append(node)
            layers[layer_num - 1][layer_ind] = node

        new_genome.layers = [new_genome.inputs, *layers, new_genome.outputs]

        new_genome.nodes = nodes
        for from_node_reduced, to_node_reduced, weight, innov in edges:
            from_layer_num, from_layer_ind, _, _, _ = from_node_reduced
            from_node = new_genome.layers[from_layer_num][from_layer_ind]
            to_layer_num, to_layer_ind, _, _, _ = to_node_reduced
            to_node = new_genome.layers[to_layer_num][to_layer_ind]
            edge = Edge(from_node, to_node, weight)
            new_genome.edges.append(edge)
            new_genome.edge_innovs.add((from_node.innov, to_node.innov))
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
            genome.add_edge(n, genome.layers[1][0])
        for n in genome.outputs:
            genome.add_edge(genome.layers[1][0], n)
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
            new_edge = Edge.copy(edge, new_genome)
            new_genome.edge_innovs.add((
                new_edge.from_node.innov,
                new_edge.to_node.innov))
        return new_genome

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
        new_node = Node(layer_num, len(self.layers[layer_num]),
                        sample_weight(self.weight_low, self.weight_high))
        self.layers[layer_num].append(new_node)
        if self.nodes and new_node.innov > self.nodes[-1].innov:
            self.nodes.append(new_node)
        else:
            self.nodes = [n for n in self.nodes if n.innov < new_node.innov] + [new_node] \
                + [n for n  in self.nodes if n.innov > new_node.innov]
        # self.node_innovs.add(node.innov)
        return new_node

    def add_edge(self, from_node, to_node):
        if (from_node.innov, to_node.innov) in self.edge_innovs:
            return
        edge = Edge(
            from_node,
            to_node,
            sample_weight(self.weight_low, self.weight_high))
        # NOTE: An error was occuring here because occasionally a from and to node pair are already an edge but also not
        # currently members of the genome in question. In which case that edge is drawn from registry and inserted in
        # the next line. This causes an error becuase that edge is likley to be a lower innov number than the one
        # before it.

        # TODO: There may be a better way here?

        if self.edges and edge.innov > self.edges[-1].innov:
            self.edges.append(edge)
        else:
            self.edges = [e for e in self.edges if e.innov < edge.innov] + [edge] \
                + [e for e in self.edges if e.innov > edge.innov]
        self.edge_innovs.add((from_node.innov, to_node.innov))
        return edge

    def __repr__(self):
        return f'Genome(edges:{len(self.edges)}, nodes:{len(self.nodes)}, fitness:{self.fitness})'

    @property
    def to_reduced_repr(self):
        all_nodes = [*self.inputs, *self.outputs, *self.nodes]
        return [node.to_reduced_repr for node in all_nodes], \
            [edge.to_reduced_repr for edge in self.edges]
