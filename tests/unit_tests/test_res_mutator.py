
from unittest.mock import patch
import unittest
from src.genome.edge import Edge
from src.genome.node import Node
from tests.unit_tests.factories import genome_factory
import itertools
from src.RES.mutator import RESMutator
from src.RES.population import RESPopulation
from src.populations.genome_seeders import curry_genome_seeder
import numpy as np
from numpy.random import normal
from src import Genome
from tests.unit_tests.factories import setup_simple_res_env


class TestMutatorClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_res_mutator_incorrect_mu_init(self):
        """Test RES mutator throws error when initial Mu value is incorrect."""
        init_mu = []
        with self.assertRaises(ValueError):
            RESMutator(
                initial_mu=init_mu,
                std_dev=0.1)

        with self.assertRaises(ValueError):
            RESMutator(
                initial_mu=0,
                std_dev=0.1)

    def test_res_mutator_incorrect_std_dev_init(self):
        """Test genomes are correctly combined when mated."""
        g1 = genome_factory()
        init_mu = [0 for _ in range(len(g1.edges) + len(g1.nodes))]
        with self.assertRaises(ValueError):
            RESMutator(
                initial_mu=init_mu,
                std_dev=[0.1, 0.2])

    def test_res_mutator_on_genome(self):
        """Test RES mutator correctly works on genome."""
        genome = genome_factory()
        init_mu = [0 for _ in range(len(genome.edges) + len(genome.nodes))]
        mutator = RESMutator(
            initial_mu=init_mu,
            std_dev=0.1)
        update = normal(init_mu, 0.1)
        with patch('numpy.random.normal',
                   side_effect=[update]):
            mutator(genome)
        _, updated_weights = genome.values()
        for a, b in zip(update, updated_weights):
            self.assertEqual(a, b)

    def test_res_mutator_on_population(self):
        """Test RES mutator correctly works on population."""

        population, mutator, init_mu = setup_simple_res_env()
        target_mu = np.random.uniform(-3, 3, len(init_mu))

        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness

        population.rank_transform()
        derivative = mutator.compute_derivative(population.genomes)
        target_direction = target_mu - init_mu

        test_value = np.dot(target_direction, derivative) / \
            (np.linalg.norm(target_direction)*np.linalg.norm(derivative))
        self.assertAlmostEqual(test_value, 1, 2)
