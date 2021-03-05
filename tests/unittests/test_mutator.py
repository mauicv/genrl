import unittest
from unittest.mock import patch
from src.genome import Genome
from src.edge import Edge
from src.node import Node
from src.mutator import Mutator
from tests.unittests.factories import genome_pair_factory
import itertools
from random import random


class TestMutatorClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_genome_weight_mutation(self):
        """Test mutator acts correctly on genomes weights."""

        NEW_UNIFORM_WEIGHT = 0.5
        with patch('numpy.random.uniform',
                   side_effect=[0.1, [0.95, 0.4, 0.95,
                                *[random() for _ in range(10)]],
                                NEW_UNIFORM_WEIGHT,
                                *[random() for _ in range(10)]]):
            with patch('numpy.random.normal',
                       side_effect=[[0.1], [0.4],
                                    [random() for _ in range(10)]]):
                g = Genome.default(input_size=2, output_size=3, depth=5)
                m_g = Genome.copy(g)
                m = Mutator()
                m.mutate_weights(m_g)

        self.assertEqual(m_g.edges[0].weight, NEW_UNIFORM_WEIGHT)
        self.assertEqual(g.edges[1].weight + 0.1, m_g.edges[1].weight)

    def test_add_node_mutation(self):
        """Test mutator correctly adds nodes to genome."""

        g = Genome.default(input_size=2, output_size=3, depth=5)
        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        m = Mutator()
        m.add_node(g)
        self.assertEqual(len(g.nodes), num_of_nodes + 1)
        self.assertEqual(len(g.edges), num_of_edges + 2)
        old_edge = [e for e in g.edges if e.disabled][0]
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
        """Test mutator correctly adds nodes to genome."""

        g = Genome.default(input_size=2, output_size=3, depth=5)
        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        greatest_edge_innov = sorted(g.edges, key=lambda n: n.innov)[-1].innov
        m = Mutator()
        m.add_edge(g)
        self.assertEqual(len(g.nodes), num_of_nodes)
        self.assertEqual(len(g.edges), num_of_edges + 1)
        new_edge = sorted(g.edges, key=lambda n: n.innov)[-1]
        self.assertEqual(new_edge.innov, greatest_edge_innov + 1)
        self.assertGreater(
            new_edge.to_node.layer_num,
            new_edge.from_node.layer_num)

    def test_mate_mutation(self):
        """Test genomes are correctly combined when mated."""

        g1, g2 = genome_pair_factory()
        for edge in g1.edges:
            edge.weight = 1
        for edge in g2.edges:
            edge.weight = -1

        m = Mutator()
        child_g = m.mate(primary=g1, secondary=g2)

        g1_nodes, g1_edges = g1.to_reduced_repr
        # g2_nodes, g2_edges = g2.to_reduced_repr
        child_g_nodes, child_g_edges = child_g.to_reduced_repr

        self.assertEqual(g1_nodes, child_g_nodes)
        self.assertEqual(
            [e[-1] for e in g1_edges],
            [e[-1] for e in child_g_edges]
        )
        self.assertEqual(
            {-1, 1},
            {e[-2] for e in child_g_edges}
        )
