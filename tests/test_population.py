import unittest
from graph.edge import Edge
from graph.node import Node
from graph.population import Population
from graph.mutator import Mutator
import itertools
from tests.factories import GenomePairFactory


class TestPopulationClass(unittest.TestCase):
    """Test methods assoicated to Node class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_compatibility_distance(self):
        """Test population."""
        g1, g2 = GenomePairFactory()
        p = Population(None)
        n1, e1 = g1.to_reduced_repr
        n2, e2 = g2.to_reduced_repr
        d = p.compare(g1, g2)
        self.assertEqual(round(d, 5), 1.7000)

    def test_populate(self):
        m = Mutator()
        p = Population(mutator=m)
        p.populate()
        self.assertEqual(len(p.population), p.population_size)
