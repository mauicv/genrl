import unittest
from src.genome import Genome
from src.edge import Edge
from src.node import Node
import itertools


class TestEdgeClass(unittest.TestCase):
    """Test methods assoicated to Genome class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_edge_init(self):
        """Test edge init."""

        n = Node(0, 0, 0)
        m = Node(0, 1, 0)
        with self.assertRaises(Exception) as context:
            Edge(n, m, 0)

        err_msg = 'Cannot connect edge to lower or same layer'
        self.assertTrue(err_msg == str(context.exception))

        k = Node(1, 0, 0)
        e = Edge(n, k, 0)
        self.assertEqual(e.from_node, n)
        self.assertEqual(e.to_node, k)

    def test_edge_copy(self):
        """Test edge can be copied onto a new genome."""

        g1 = Genome.default(input_size=1, output_size=1, depth=1)
        g2 = Genome.default(input_size=1, output_size=1, depth=1)

        n = g1.add_node(1)
        g2.add_node(1)

        e1 = g1.add_edge(g1.layers[0][0], n)
        g1.add_edge(n, g1.layers[2][0])

        mutation_rate = 0.1
        ne = Edge.copy(e1, g2)

        self.assertNotEqual(ne, e1)
        self.assertEqual(ne.innov, e1.innov)
        self.assertLess(abs(ne.weight - e1.weight), mutation_rate)

    def test_edge_copy_err(self):
        """Test copy edge throws correct error if node doesn't exist on
        new_genome."""
        g1 = Genome.default(input_size=1, output_size=1, depth=1)
        g2 = Genome.default(input_size=1, output_size=1, depth=1)

        n = g1.add_node(1)

        e1 = g1.add_edge(g1.layers[0][0], n)
        g1.add_edge(n, g1.layers[2][0])

        with self.assertRaises(Exception) as context:
            Edge.copy(e1, g2)

        err_msg = 'to_node does not exist on new_genome.'
        self.assertTrue(err_msg == str(context.exception))
