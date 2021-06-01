import unittest
from unittest.mock import patch
from gerel.genome.factories import minimal
from gerel.genome.edge import Edge
from gerel.genome.node import Node
from gerel.algorithms.NEAT.functions import add_node, add_edge, curry_weight_mutator, curry_crossover
from tests.factories import genome_pair_factory
from gerel.genome.factories import copy
import itertools
from random import random


class TestMutatorClass(unittest.TestCase):
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    @unittest.skip("stochastic element needs removed")
    def test_genome_weight_mutation(self):
        new_uniform_weight = 0.5
        with patch('numpy.random.uniform',
                   side_effect=[0.1, *[0.95, 0.4, 0.95,
                                *[random() for _ in range(10)]],
                                new_uniform_weight,
                                *[random() for _ in range(10)]]):
            with patch('numpy.random.normal',
                       side_effect=[[0.1], [0.4],
                                    [random() for _ in range(10)]]):
                g = minimal(input_size=2, output_size=3, depth=5)
                m_g = copy(g)
                mutate_weights = curry_weight_mutator(
                    weight_mutation_likelihood=0.8,
                    weight_mutation_rate_random=0.1,
                    weight_mutation_rate_uniform=0.9,
                    weight_mutation_variance=0.1
                )
                mutate_weights(m_g)

        self.assertEqual(m_g.edges[0].weight, new_uniform_weight)
        self.assertEqual(g.edges[1].weight + 0.1, m_g.edges[1].weight)

    def test_add_node_mutation(self):
        g = minimal(input_size=2, output_size=3, depth=5)
        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        add_node(g)
        self.assertEqual(len(g.nodes), num_of_nodes + 1)
        self.assertEqual(len(g.edges), num_of_edges + 2)
        old_edge = [e for e in g.edges if not e.active][0]
        new_node = sorted(g.nodes, key=lambda n: n.innov)[-1]
        self.assertEqual(
            new_node.edges_in[0].from_node.layer_num,
            old_edge.from_node.layer_num
        )
        self.assertEqual(
            new_node.edges_out[0].to_node.layer_num,
            old_edge.to_node.layer_num
        )

    @unittest.skip("stochastic element needs removed")
    def test_add_edge_mutation(self):
        g = minimal(input_size=2, output_size=3, depth=5)
        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        greatest_edge_innov = sorted(g.edges, key=lambda n: n.innov)[-1].innov
        add_edge(g)
        self.assertEqual(len(g.nodes), num_of_nodes)
        self.assertEqual(len(g.edges), num_of_edges + 1)
        new_edge = sorted(g.edges, key=lambda n: n.innov)[-1]
        self.assertEqual(new_edge.innov, greatest_edge_innov + 1)
        self.assertGreater(
            new_edge.to_node.layer_num,
            new_edge.from_node.layer_num)

    def test_mate_mutation(self):
        g1, g2 = genome_pair_factory()
        for edge in g1.edges:
            edge.weight = 1
        for edge in g2.edges:
            edge.weight = -1

        crossover_fn = curry_crossover(gene_disable_rate=0.75)
        child_g = crossover_fn(primary=g1, secondary=g2)

        g1_nodes, g1_edges = g1.to_reduced_repr
        child_g_nodes, child_g_edges = child_g.to_reduced_repr

        self.assertEqual(g1_nodes, child_g_nodes)
        self.assertEqual(
            [e[-1] for e in g1_edges],
            [e[-1] for e in child_g_edges]
        )
        self.assertEqual(
            {-1, 1},
            {e[-3] for e in child_g_edges}
        )

    # TODO: fix
    @unittest.skip("stochastic element needs removed")
    def test_mate_mutation_disabled_edges(self):
        g1, g2 = genome_pair_factory()
        for edge in g1.edges:
            edge.weight = 1
        for edge in g2.edges:
            edge.weight = -1

        crossover_fn = curry_crossover(gene_disable_rate=0.75)
        child_g = crossover_fn(primary=g1, secondary=g2)
