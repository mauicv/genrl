import numpy as np
import unittest
from unittest.mock import Mock
from graph.genome import Genome, Edge, Node
from graph.mutator import Mutator
import itertools
from random import random

class TestMutatorClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()

    def test_genome_mutation(self):
        """Test mutator acts correctly on genomes."""

        NEW_UNIFORM_WEIGHT = 0.5
        np.random.uniform = Mock(side_effect =
            [0.1,
             [0.95, 0.4, 0.95, *[random() for _ in range(10)]],
             NEW_UNIFORM_WEIGHT,
             *[random() for _ in range(10)]
        ])
        np.random.normal = Mock(side_effect = [[0.1], [0.4],
                                [random() for _ in range(10)]])

        g = Genome.default(input_size=2, output_size=3, depth=5)
        m = Mutator()
        m_g = m.mutate(g)

        self.assertEqual(m_g.edges[0].weight, 0.5)
        self.assertEqual(g.edges[1].weight + 0.1, m_g.edges[1].weight)
