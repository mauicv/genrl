from src.genome.genome import Genome
from src.NEAT.mutator import NEATMutator


def curry_genome_seeder(mutator=NEATMutator()):
    def genome_seeder(n, seed_genomes=None):
        if not seed_genomes:
            seed_genomes = [Genome.default()]

        count = 0
        while True:
            for seed_genome in seed_genomes:
                genome = Genome.copy(seed_genome)
                mutator(genome)
                yield genome
                count += 1
                if count == n:
                    break
    return genome_seeder
