import unittest
from src.genome.genome import Genome
from src.genome.edge import Edge
from src.genome.node import Node
import itertools


class TestGenomeClass(unittest.TestCase):
    """Test methods assoicated to Genome class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_genome_init(self):
        """Test genome __init__ function.

        New instance of genome contains specified number of input and output
        nodes plus a single connecting node in the middle.
        """
        g = Genome.default(input_size=2, output_size=3, depth=5)

        # Check correct nodes
        self.assertEqual(len(g.inputs), 2)
        self.assertEqual(len(g.nodes), 1)
        self.assertEqual(len(g.outputs), 3)
        for node in g.inputs:
            self.assertEqual(node.layer_num, 0)
        for node in g.outputs:
            self.assertEqual(node.layer_num, len(g.layers) - 1)
        self.assertEqual(len(g.layers), 7)

        # check central connecting node edges
        self.assertEqual(
            g.layer_edges_in(layer_num=6),
            g.layer_edges_out(layer_num=1)
        )
        self.assertEqual(
            g.layer_edges_in(layer_num=1),
            g.layer_edges_out(layer_num=0)
        )

    def test_get_addmissable_edges(self):
        g = Genome
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])

        self.assertEqual(len(g.layer_edges_in(3)), 1)
        self.assertEqual(len(g.layer_edges_out(3)), 1)

        for layer_num in [2, 4, 5]:
            self.assertEqual(len(g.layer_edges_in(layer_num)), 0)
            self.assertEqual(len(g.layer_edges_out(layer_num)), 0)

        self.assertEqual(len(g.get_addmissable_edges()), 5)
        addmissable = lambda e: e.to_node.layer_num - e.from_node.layer_num > 1
        for e in g.get_addmissable_edges():
            self.assertEqual(addmissable(e), True)

    def test_copy_genome(self):
        g = Genome
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])
        g_copy = Genome.copy(g)
        self.assertNotEqual(g_copy, g)
        for node_copy, node in zip(g_copy.nodes, g.nodes):
            self.assertEqual(node_copy.innov, node.innov)

        for layer_copy, layer in zip(g_copy.layers, g.layers):
            self.assertEqual(len(layer_copy), len(layer))
            for node_copy, node in zip(layer_copy, layer):
                self.assertNotEqual(node_copy, node)
                self.assertEqual(node_copy.innov, node.innov)
                self.assertEqual(len(node_copy.edges_in), len(node.edges_in))
                self.assertEqual(len(node_copy.edges_out), len(node.edges_out))

        for edge_copy, edge in zip(g_copy.edges, g.edges):
            self.assertNotEqual(edge_copy, edge)
            self.assertEqual(edge_copy.innov, edge.innov)

    def test_from_genes_constructor(self):
        g = Genome
        n2 = g.add_node(4)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][1], n2)
        g.add_edge(n2, g.outputs[2])
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[2])

        edges = [e.to_reduced_repr for e in g.edges]
        nodes = [n.to_reduced_repr for n in g.nodes]
        g_2 = Genome.from_genes(
            nodes,
            edges,
            input_size=2,
            output_size=3,
            depth=5)

        self.assertEqual(
            [[n.to_reduced_repr for n in layer] for layer in g_2.layers],
            [[n.to_reduced_repr for n in layer] for layer in g.layers]
        )

        self.assertEqual(
            [edge.to_reduced_repr for edge in g_2.edges],
            [edge.to_reduced_repr for edge in g.edges]
        )