import unittest
from src.genome.edge import Edge
from src.genome.node import Node
import itertools
from src.RES.mutator import RESMutator
from src.RES.population import RESPopulation
import numpy as np
from numpy.random import normal
from tests.unit_tests.factories import setup_simple_res_env


class TestRESPopulationClass(unittest.TestCase):
    """Test methods assoicated to Mutator class."""

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_res_rank_ordering(self):
        """Test RES mutator correctly works on population."""

        population, mutator, init_mu = setup_simple_res_env()
        target_mu = np.random.uniform(-3, 3, len(init_mu))

        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness

        ordered_genomes_1 = sorted(population.genomes, key=lambda item: item.fitness)

        population.rank_transform()
        for genome in population.genomes:
            self.assertLessEqual(genome.fitness, 0.5)
            self.assertGreaterEqual(genome.fitness, -0.5)

        ordered_genomes_2 = sorted(population.genomes, key=lambda item: item.fitness)

        for genome_1, genome_2 in zip(ordered_genomes_1, ordered_genomes_2):
            self.assertEqual(genome_1, genome_2)

        for genome1, genome_2 in zip(ordered_genomes_1, ordered_genomes_1[1:]):
            fitness_diff = genome_2.fitness - genome1.fitness
            self.assertAlmostEqual(fitness_diff, 1/population.population_size, 5)
