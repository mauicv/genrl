import unittest
from src.genome.edge import Edge
from src.genome.node import Node
import itertools
from tests.unit_tests.factories import genome_pair_factory
from src import generate_neat_metric


class TestMetrics(unittest.TestCase):
    """Test metrics."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_compatibility_distance(self):
        """Test neat metric."""
        metric = generate_neat_metric()
        g1, g2 = genome_pair_factory()
        # n1, e1 = g1.to_reduced_repr
        # n2, e2 = g2.to_reduced_repr
        d = metric(g1, g2)
        self.assertEqual(round(d, 5), 0.9)
