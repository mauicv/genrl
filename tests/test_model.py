import unittest
from random import random
from src.util import sample_weight
from tests.factories import genome_factory
from src.edge import Edge
from src.node import Node
from src.model import Model
import itertools


class TestModelClass(unittest.TestCase):
    """Test methods assoicated to Genome class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_model_init(self):
        """Test model __init__ function."""
        def weight_gen():
            while True:
                yield sample_weight(-2, 2)
        def bias_gen():
            while True:
                yield sample_weight(-2, 2)

        g = genome_factory(weight_gen=weight_gen(), bias_gen=bias_gen())
        model = Model(g.to_reduced_repr)
        nodes, edges = g.to_reduced_repr
        for (n1i, n1j, _, n1w), (n2i, n2j, _, n2w), ew, _ in edges:
            layer = model.layers[n1i]
            for cell in layer.inputs:
                if (cell.i, cell.j) == (n1i, n1j):
                    from_cell = cell
                    self.assertEqual(n1w, from_cell.b)
            for cell in layer.outputs:
                if (cell.i, cell.j) == (n2i, n2j):
                    to_cell = cell
                    self.assertEqual(n2w, to_cell.b)
            self.assertIsNotNone(from_cell)
            self.assertIsNotNone(to_cell)
            i = layer.inputs.index(from_cell)
            j = layer.outputs.index(to_cell)
            self.assertEqual(layer.mat[i, j], ew)