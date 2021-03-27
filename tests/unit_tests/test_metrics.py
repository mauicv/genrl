import unittest
from gerel.genome.edge import Edge
from gerel.genome.node import Node
import itertools
from tests.unit_tests.factories import genome_pair_factory
from gerel.algorithms.NEAT.metric import generate_neat_metric


class TestMetrics(unittest.TestCase):
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_compatibility_distance(self):
        metric = generate_neat_metric()
        g1, g2 = genome_pair_factory()
        d = metric(g1, g2)
        self.assertEqual(round(d, 5), 0.9)
