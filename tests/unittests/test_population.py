import unittest
from src.edge import Edge
from src.node import Node
from src.population import Population
from src.mutator import Mutator
import itertools
from tests.unittests.factories import genome_pair_factory


class TestPopulationClass(unittest.TestCase):
    """Test methods assoicated to Node class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_populate(self):
        m = Mutator()
        p = Population(mutator=m)
        p.populate()
        self.assertEqual(len(p.genomes), p.population_size)

    def test_sort(self):
        self.assertEqual(1, 0)
