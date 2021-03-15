from src.genome.genome import Genome


def curry_genome_seeder(mutator):
    def genome_seeder(n, seed_genomes=None):
        if not seed_genomes:
            seed_genomes = [Genome.default()]

        count = 0
        while count < n:
            for seed_genome in seed_genomes:
                genome = Genome.copy(seed_genome)
                mutator(genome)
                yield genome
                count += 1
    return genome_seeder
