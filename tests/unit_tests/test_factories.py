import unittest
from src.genome.edge import Edge
from src.genome.node import Node
import itertools
from src.genome.factories import dense, minimal, copy, from_genes
import numpy as np


class TestGenomeFactories(unittest.TestCase):
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_dense_factory(self):
        from src.algorithms.RES.population import RESPopulation
        from src.algorithms.RES.mutator import RESMutator
        from src.populations.genome_seeders import curry_genome_seeder

        genome = dense(
            input_size=1,
            output_size=1,
            layer_dims=[4]
        )

        weights_len = len(genome.edges) + len(genome.nodes)
        init_mu = np.random.uniform(-1, 1, weights_len)

        mutator = RESMutator(
            initial_mu=init_mu,
            std_dev=0.1,
            alpha=0.1
        )

        seeder = curry_genome_seeder(
            mutator=mutator,
            seed_genomes=[genome]
        )

        RESPopulation(
            population_size=1,
            genome_seeder=seeder
        )

    def test_minimal_factory(self):
        g = minimal(input_size=2, output_size=3, depth=5)

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

    def test_copy_factory(self):
        g = minimal()
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])
        g_copy = copy(g)
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
        g = minimal(input_size=2, output_size=3, depth=5)
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
        g_2 = from_genes(
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
