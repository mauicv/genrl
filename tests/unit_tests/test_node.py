import unittest
from src.genome.edge import Edge
from src.genome.node import Node
import itertools


class TestNodeClass(unittest.TestCase):
    """Test methods assoicated to Node class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_node_init(self):
        """Test node."""

        n1 = Node(0, 0, 0)
        n2 = Node(0, 1, 0)
        n3 = Node(1, 0, 0)
        n4 = Node(1, 1, 0)

        # Check correct nodes
        self.assertEqual(n1.innov, 0)
        self.assertEqual(n2.innov, 1)
        self.assertEqual(n3.innov, 2)
        self.assertEqual(n4.innov, 3)

    def test_node_copy(self):
        """Test node."""
        n = Node(0, 0, 0)
        m = Node.copy(n)
        self.assertNotEqual(n, m)
        self.assertEqual(n.innov, m.innov)
        self.assertEqual(n.layer_num, m.layer_num)
        self.assertEqual(n.layer_ind, m.layer_ind)
