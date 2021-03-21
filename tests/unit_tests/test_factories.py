import unittest
from src.genome.edge import Edge
from src.genome.node import Node
import itertools
from src.genome.factories import dense
import numpy as np
from src.util import print_genome


class TestGenomeFactories(unittest.TestCase):
    """Test methods assoicated to Genome class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_dense(self):
        from src.RES.population import RESPopulation
        from src.RES.mutator import RESMutator
        from src.populations.genome_seeders import curry_genome_seeder

        genome = dense(
            input_size=1,
            output_size=1,
            layer_dims=[4]
        )

        weights_len = len(genome.edges) + len(genome.nodes)
        init_mu = np.random.uniform(-1, 1, weights_len)

        mutator = RESMutator(
            initial_mu=init_mu,
            std_dev=0.1,
            alpha=0.1
        )

        seeder = curry_genome_seeder(
            mutator=mutator,
            seed_genomes=[genome]
        )

        RESPopulation(
            population_size=1,
            genome_seeder=seeder
        )
