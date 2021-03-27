from time import time
from gerel.debug.class_debug_decorator import add_inst_validator
from gerel.debug.population_validator import validate_population
import numpy as np
from inspect import isgeneratorfunction


@add_inst_validator(env="TESTING", validator=validate_population)
class Population:
    def __init__(
            self,
            genome_seeder,
            population_size,
            delta=None,
            metric=None,
    ):
        """Construct Population of genomes.

        Stores a collection of genomes and provides methods for mutating them.

        :param genome_seeder: An iterable of genomes that will be used to generate the
            initial population.
        :param population_size: An integer number giving the size of the total Population.
        :param delta: A float number that defines the minimum distance two genomes require
            to be apart before there sorted into different species. Set to None if
            single species is desired (no speciation).
        :param metric: Function that takes two genomes and returns the distance between them.
            The metric is used to partition the genomes into species. If None then no
            speciation will take place.
        :raises ValueError: If delta is not equal to None but metric is.
        """
        if delta and not metric:
            raise ValueError('Speciation requires a metric')

        self.population_size = population_size
        self.delta = delta
        self.species = {}
        self.genomes = []
        self.generation = 0
        self.metric = metric
        genome_seeder = genome_seeder(population_size) \
            if isgeneratorfunction(genome_seeder) else genome_seeder
        self.genomes = [genome for genome in genome_seeder]

    def speciate(self):
        """Divides the population of genomes up into species.

        :return: None
        """
        self.species[1] = {
            'repr': self.genomes[0],
            'group': [self.genomes[0]]
        }

        for genome in self.genomes[1:]:
            assigned_group = False
            for key, item in self.species.items():
                if self.delta is None or self.metric is None or \
                        self.metric(genome, item['repr']) < self.delta:
                    assigned_group = True
                    self.species[key]['group'].append(genome)
                    break
            if not assigned_group:
                self.species[len(self.species) + 1] = {
                    'repr': genome,
                    'group': [genome]
                }

        for key, item in self.species.items():
            group_size = len(item['group'])
            adj_fitness = lambda x: x.fitness / group_size
            group_fitness = sum([adj_fitness(g) for g in item['group']])
            item['group_fitness'] = group_fitness
            item['group'].sort(key=adj_fitness, reverse=True)
            item['generation'] = self.generation

    def to_dict(self):
        """Returns dictionary summary of population.

        :return: dictionary summary of population.
        """
        best_genome = max(self.genomes, key=lambda g: g.fitness)
        worst_genome = min(self.genomes, key=lambda g: g.fitness)
        data = {
            'generation': self.generation,
            'species_num': len(self.species),
            'best_fitness': best_genome.fitness,
            'best_genome': best_genome.to_reduced_repr,
            'worst_fitness': worst_genome.fitness,
            'worst_genome': worst_genome.to_reduced_repr,
            'mean_fitness': np.mean([g.fitness for g in self.genomes]),
            'groups': [],
            'created_at': time()
        }
        if self.metric is not None:
            for key, item in self.species.items():
                if item['generation'] != self.generation:
                    raise ValueError('current generation and species group generation are not the same!')
                data['groups'].append({
                    'id': key,
                    'bests': [g.to_reduced_repr for g in item['group'][0:3]],
                    'group_fitness': item['group_fitness'],
                    'repr': item['repr'].to_reduced_repr,
                    'partition': {
                        key: self.metric(item['repr'], other_item['repr'])
                        for key, other_item in self.species.items()
                    }
                })
        return data
