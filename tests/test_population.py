import unittest
from src.edge import Edge
from src.node import Node
from src.population import Population
from src.mutator import Mutator
import itertools
from tests.factories import genome_pair_factory


class TestPopulationClass(unittest.TestCase):
    """Test methods assoicated to Node class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_compatibility_distance(self):
        """Test population."""
        g1, g2 = genome_pair_factory()
        p = Population(None)
        n1, e1 = g1.to_reduced_repr
        n2, e2 = g2.to_reduced_repr
        d = p.compare(g1, g2)
        self.assertEqual(round(d, 5), 0.9)

    def test_populate(self):
        m = Mutator()
        p = Population(mutator=m)
        p.populate()
        self.assertEqual(len(p.population), p.population_size)
