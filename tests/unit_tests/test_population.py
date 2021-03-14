import unittest
from src.genome import Genome
from src.population import Population
from src.mutator import Mutator
from src.metrics import generate_neat_metric
from random import random, choice
from src.node import Node
from src.edge import Edge
import itertools
from src.util import equal_genome


class TestPopulationClass(unittest.TestCase):
    """Test methods associated to Population class."""
    def setUp(self):
        # reset innovation number
        Node.innov_iter = itertools.count()
        Edge.innov_iter = itertools.count()
        Node.registry = {}
        Edge.registry = {}

    def test_populate(self):
        m = Mutator()
        g = Genome.default()
        p = Population(population_size=50, seed_genomes=[g], mutator=m)
        for genome in p.genomes:
            if len(set(e.innov for e in genome.edges)) != len(genome.edges):
                print(Edge.registry)
                print(genome.edge_innovs)
                for e in genome.edges:
                    print(e.to_reduced_repr)
            self.assertEqual(len(set(n.innov for n in genome.nodes)), len(genome.nodes))
            self.assertEqual(len(set(e.innov for e in genome.edges)), len(genome.edges))
        self.assertEqual(len(p.genomes), p.population_size)

    def test_evolve(self):
        pop_size = 30
        m = Mutator()
        g = Genome.default()
        p = Population(
            population_size=pop_size,
            seed_genomes=[g],
            mutator=m,
            delta=1
        )
        d = generate_neat_metric()

        for i in range(10):
            # the following for loop is a simulation of the env role out and fitness computation.
            for g in p.genomes:
                g.fitness = random()
            p.step(d)

        for key, item in p.species.items():
            self.assertEqual(len(set(item['group'])), len(item['group']))
            test_genome = choice(item['group'])
            metric_dist = d(test_genome, item['repr'])
            self.assertLess(metric_dist, p.delta)
            if metric_dist == 0:
                self.assertTrue(equal_genome(test_genome, item['repr']))
