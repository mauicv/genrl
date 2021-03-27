import unittest
from gerel.genome.genome import DimensionMismatchError
from gerel.genome.edge import Edge
from gerel.genome.node import Node
from random import random
import itertools
from gerel.genome.factories import minimal


class TestGenomeClass(unittest.TestCase):
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_get_addmissable_edges(self):
        g = minimal(input_size=2, output_size=3, depth=5)
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])

        self.assertEqual(len(g.layer_edges_in(3)), 1)
        self.assertEqual(len(g.layer_edges_out(3)), 1)

        for layer_num in [2, 4, 5]:
            self.assertEqual(len(g.layer_edges_in(layer_num)), 0)
            self.assertEqual(len(g.layer_edges_out(layer_num)), 0)

        self.assertEqual(len(g.get_admissible_edges()), 5)
        addmissable = lambda e: e.to_node.layer_num - e.from_node.layer_num > 1
        for e in g.get_admissible_edges():
            self.assertEqual(addmissable(e), True)

    def test_update_weights_error(self):
        g = minimal(input_size=2, output_size=3, depth=5)
        with self.assertRaises(DimensionMismatchError):
            g.weights = [0 for _ in range(len(g.edges) + len(g.nodes) - 1)]

    def test_update_weights(self):
        g = minimal(input_size=2, output_size=3, depth=5)
        update_vector = [random() for _ in range(len(g.edges) + len(g.nodes))]
        g.weights = update_vector
        updated_weights = [target.weight for target in itertools.chain(g.nodes, g.edges)]
        self.assertEqual(update_vector, updated_weights)
