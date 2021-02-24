"""Population Class.

The distance measure δ allows us to speciate using a compatibility threshold
δt. An ordered list of species is maintained. In each generation, genomes are
sequentially placed into species.

# TODO:
    - compatibility distance
    - speciation
    - explicit fitness sharing


Compatibility Distance:

δ = c_1*E/N + c_2*D/N + c3 · W.

where E is excess and D is disjoint genes and W is the average weight
differences of matching genes including disabled genes

"""

POPULATION                      = 150
C_1                             = 1.0
C_2                             = 1.0
C_3                             = 0.4
DELTA                           = 3.0
MUTATION_WITHOUT_CROSSOVER_RATE = 0.25
INSTERSPECIES_MATING_RATE       = 0.001


class Population:
    def __init__(
            self,
            c_1=C_1,
            c_2=C_2,
            c_3=C_3,
            delta=DELTA,
            mutation_without_crossover_rate=MUTATION_WITHOUT_CROSSOVER_RATE,
            insterspecies_mating_rate=INSTERSPECIES_MATING_RATE):
        self.c_1 = c_1
        self.c_2 = c_2
        self.c_3 = c_3
        self.delta = delta
        self.mutation_without_crossover_rate = mutation_without_crossover_rate
        self.insterspecies_mating_rate = insterspecies_mating_rate
        self.centers = []
        self.species = {}
        self.population = []

    def compare(self, genome_1, genome_2):
        self.compare_gene_difference(genome_1.nodes, genome_2.nodes)
        self.compare_gene_difference(genome_1.edges, genome_2.edges)

    def compare_gene_difference(self, genes_1, genes_2):
        """Compatibility Distance:

            δ = c_1*E/N + c_2*D/N + c3 · W.

        where E is excess and D is disjoint genes and W is the average weight
        differences of matching genes including disabled genes."""

        last_match = [0, 0]
        W, M, i, j = (0, 0, 0, 0)
        while True:
            if genes_1[i].innov == genes_2[j].innov:
                last_match = [i, j]
                W += abs(genes_1[i].weight - genes_1[i].weight)
                M, i, j = (M + 1, i + 1, j + 1)
            elif genes_1[i].innov < genes_2[j].innov:
                i += 1
            elif genes_1[i].innov > genes_2[j].innov:
                j += 1
            if i == len(genes_1) or j == len(genes_2):
                break
        N = max(len(genes_1), len(genes_2))
        D = last_match[0] + last_match[1] - 2 * M
        E = len(genes_1) - last_match[0] + len(genes_2) - last_match[1]
        W = W / M
        return self.c_1 * E / N + self.c_2 * D / N + self.c_3 * W

    def sort(self):
        pass
