import unittest
from src.genome.edge import Edge
from src.genome.node import Node
from tests.unit_tests.factories import genome_factory
import itertools
from src.algorithms import SIMPLEMutator
import numpy as np
from numpy.random import normal
from tests.unit_tests.factories import setup_simple_res_env


class TestSIMPLEMutatorClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_simple_mutator_on_genome(self):
        """Test SIMPLE mutator correctly works on genome."""
        genome = genome_factory()
        mutator = SIMPLEMutator(
            std_dev=0.01,
            survival_rate=0.1)
        old_weights = genome.weights
        mutator(genome)
        updated_weights = genome.weights
        difference = np.linalg.norm(np.array(old_weights) - np.array(updated_weights))
        self.assertLess(difference, 0.06)

    def test_simple_mutator_on_population(self):
        """Test SIMPLE mutator correctly works on population.

        TODO: hard to test...
        """

        population, mutator, init_mu = \
            setup_simple_res_env(mutator_type=SIMPLEMutator)
        target_mu = np.random.uniform(-3, 3, len(init_mu))

        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness

        max_genome = max(population.genomes, key=lambda g: g.fitness)
        min_genome = min(population.genomes, key=lambda g: g.fitness)
        mutator(population)
        self.assertIn(max_genome, population.genomes)
        self.assertNotIn(min_genome, population.genomes)
        self.assertEqual(len(population.genomes), population.population_size)
