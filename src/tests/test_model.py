from gerel.util.activations import step
from gerel.util.util import sample_weight
import unittest
from tests.factories import genome_factory
from gerel.genome.edge import Edge
from gerel.genome.node import Node
from gerel.model.model import Model
import itertools
from gerel.util.activations import build_leaky_relu, tanh


class TestModelClass(unittest.TestCase):
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_model_init(self):
        def weight_gen():
            while True:
                yield sample_weight(-2, 2)

        def bias_gen():
            while True:
                yield sample_weight(-2, 2)

        g = genome_factory(weight_gen=weight_gen(), bias_gen=bias_gen())
        model = Model(g.to_reduced_repr)
        nodes, edges = g.to_reduced_repr
        for (n1i, n1j, _, n1w, _), (n2i, n2j, _, n2w, _), ew, _, active in edges:
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

    # @unittest.skip("Reduced rep is inccorect")
    def test_model_run(self):
        # TODO: replace in below ds.
        # in1 = (0, 0, -1, 0, 'input')
        # in2 = (0, 1, -1, 0, 'input')
        # l1n1 = (1, 0, 1, 1, 'hidden')
        # l1n2 = (1, 1, 2, 0, 'hidden')
        # l1n3 = (1, 2, 3, -1, 'hidden')
        # l2n1 = (2, 0, 4, 0, 'hidden')
        # l2n2 = (2, 1, 5, 1, 'hidden')
        # l3n1 = (3, 0, -1, -1, 'output')
        # l3n2 = (3, 1, -1, 0, 'output')

        model = Model(([
                            (0, 0, -1, 0, 'input'), (0, 1, -1, 0, 'input'),
                            (1, 0, 1, 1, 'hidden'), (1, 1, 2, 0, 'hidden'), (1, 2, 3, -1, 'hidden'),
                            (2, 0, 4, 0, 'hidden'), (2, 1, 5, 1, 'hidden'),
                            (3, 0, -1, -1, 'output'), (3, 1, -1, 0, 'output')
                       ],
                       [
                            ((0, 0, -1, 0, '_'), (3, 0, -1, -1, '_'), 10, 0, False),
                            ((0, 0, -1, 0, '_'), (1, 0, 1, 1, '_'), 1, 0, True),
                            ((0, 0, -1, 0, '_'), (1, 1, 2, 0, '_'), -1, 1, True),
                            ((0, 1, -1, 0, '_'), (1, 1, 2, 0, '_'), -1, 2, True),
                            ((0, 1, -1, 0, '_'), (3, 1, 7, 0, '_'), 2, 3, True),
                            ((0, 1, -1, 0, '_'), (1, 2, 3, -1, '_'), 1, 4, True),
                            ((1, 0, 1, 1, '_'), (2, 0, 4, 0, '_'), 1, 5, True),
                            ((1, 1, 2, 0, '_'), (2, 0, 4, 0, '_'), 1, 6, True),
                            ((1, 1, 2, 0, '_'), (2, 1, 5, 1, '_'), -1, 7, True),
                            ((1, 2, 3, -1, '_'), (2, 1, 5, 1, '_'), 2, 8, True),
                            ((2, 0, 4, 0, '_'), (3, 0, -1, -1, '_'), 1, 9, True),
                            ((2, 0, 4, 0, '_'), (3, 1, -1, 0, '_'), -1, 10, True),
                            ((2, 1, 5, 1, '_'), (3, 0, -1, -1, '_'), 2, 11, True),
                            ((2, 1, 5, 1, '_'), (3, 1, -1, 0, '_'), -1, 12, True)
                       ]))

        # overwrite reset method so that we can inspect the run call
        model.reset = lambda: None
        self.assertEqual(model([1, 2]), [0.0, 4.0])
        for layer, target in zip(model.layers[1:], [[1, -1, 1], [-1, 1]]):
            self.assertEqual(target, [step(cell.acc + cell.b) for cell in layer.inputs])

    def test_model_reset(self):
        g = genome_factory()
        model = Model(g.to_reduced_repr)
        model([1, 2])
        for _, cell in model.cells.items():
            self.assertEqual(cell.acc, 0)

    def test_model_mismatch_err(self):
        g = genome_factory()
        model = Model(
            g.to_reduced_repr,
            layer_activations=[step for _ in range(6)]
        )
        self.assertEqual(len([layer.activation for layer in model.layers]), 6)

        with self.assertRaises(Exception) as context:
            model = Model(
                g.to_reduced_repr,
                layer_activations=[step for _ in range(5)]
            )

        MSG = 'Mismatch in number of activation functions and layers'
        self.assertEqual(MSG, str(context.exception))

    def test_model_activation_fns(self):
        g = genome_factory()
        leaky_relu = build_leaky_relu(0.1)
        model = Model(
            g.to_reduced_repr,
            layer_activations=[
                lambda x: -x,
                leaky_relu,
                lambda x: x,
                lambda x: x,
                lambda x: x,
                lambda x: x
            ]
        )
        self.assertEqual(model([2, 2]), [-1.3, -1.3, -2.3])
