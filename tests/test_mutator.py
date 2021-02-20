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

    def test_add_node_mutation(self):
        """Test mutator correctly adds nodes to genome."""

        np.random.uniform = Mock(side_effect=[0.02, 0.04])
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
