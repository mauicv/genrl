import numpy as np
import unittest
from unittest.mock import Mock
from graph.genome import Genome
from graph.edge import Edge
from graph.node import Node
from graph.mutator import Mutator
import itertools
from random import random


class TestMutatorClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_genome_weight_mutation(self):
        """Test mutator acts correctly on genomes weights."""

        NEW_UNIFORM_WEIGHT = 0.5
        np.random.uniform = Mock(
            side_effect=[0.1, [0.95, 0.4, 0.95,
                         *[random() for _ in range(10)]],
                         NEW_UNIFORM_WEIGHT,
                         *[random() for _ in range(10)]
                         ])

        np.random.normal = Mock(side_effect=[[0.1], [0.4],
                                [random() for _ in range(10)]])

        g = Genome.default(input_size=2, output_size=3, depth=5)
        m = Mutator()
        m_g = m.mutate_weights(g)

        self.assertEqual(m_g.edges[0].weight, 0.5)
        self.assertEqual(g.edges[1].weight + 0.1, m_g.edges[1].weight)

    def test_genome_topology_mutation(self):
        """Test mutator acts correctly on genomes weights."""
        np.random.uniform = Mock(side_effect=[0.02, 0.04, 0.12, 0.3])
        g = Genome.default(input_size=2, output_size=3, depth=5)
        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        m = Mutator()
        m.mutate_topology(g)

        self.assertEqual(len(g.edges), num_of_edges + 3)
        self.assertEqual(len(g.nodes), num_of_nodes + 1)

        num_of_nodes = len(g.nodes)
        num_of_edges = len(g.edges)
        m = Mutator()
        m.mutate_topology(g)

        self.assertEqual(len(g.edges), num_of_edges)
        self.assertEqual(len(g.nodes), num_of_nodes)

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

        g1 = Genome.default(input_size=2, output_size=3, depth=5)
        n1 = g1.add_node(4)
        g1.add_edge(g1.layers[0][0], n1)
        g1.add_edge(n1, g1.outputs[0])

        n2 = g1.add_node(3)
        g1.add_edge(g1.layers[0][1], n2)
        g1.add_edge(n2, g1.outputs[2])

        g2 = Genome.copy(g1)

        n4 = g2.add_node(2)
        g2.add_edge(g2.layers[0][1], n4)
        g2.add_edge(n4, g2.layers[3][0])

        n5 = g1.add_node(4)
        g1.add_edge(g1.layers[3][0], n5)
        g1.add_edge(n5, g1.outputs[0])

        n3 = g1.add_node(3)
        e1 = g1.add_edge(g1.layers[0][0], n3)
        e2 = g1.add_edge(n3, g1.outputs[2])
        g2.layers[3].append(Node.copy(n3))
        g2.nodes.append(Node.copy(n3))

        g2.add_edge(g2.layers[0][0], n3, innov=e1.innov)
        e2 = g2.add_edge(n3, g2.outputs[2], innov=e2.innov)

        print('g1: ', g1)
        print('g2: ', g2)

        m = Mutator()
        child_g = m.mate(primary=g1, secondary=g2)
        print('child_g: ', child_g)
