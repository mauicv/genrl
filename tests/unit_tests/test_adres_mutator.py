
import unittest
from gerel.genome.edge import Edge
from gerel.genome.node import Node
import itertools
from gerel.algorithms.RES.mutator import ADRESMutator
import numpy as np
from numpy.random import normal
from tests.unit_tests.factories import setup_simple_res_env


class TestADRESMutatorClass(unittest.TestCase):

    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_ADRES_mutator_on_population(self):
        population, mutator, init_mu = setup_simple_res_env(mutator_type=ADRESMutator)
        target_mu = np.random.uniform(-3, 3, len(init_mu))

        for genome in population.genomes:
            # fitness is symmetrically allocated.
            _, ws = genome.values()
            ws = np.array(ws)
            fitness = 1 / np.power(target_mu - ws, 2).sum()
            genome.fitness = fitness

        population.rank_transform()
        derivative = mutator.compute_derivative(population.genomes)
        pref = lambda g: mutator.compute_genome_preference(g, derivative)
        preferred_genome = max(population.genomes, key=pref)
        mutator.update_mu(population, derivative)
        _, new_mu = preferred_genome.values()
        self.assertEqual(mutator.mu, new_mu)
