"""Compatibility Distance:

    δ = c_1*E/N + c_2*D/N + c3 · W.

where E is excess and D is disjoint genes and W is the average weight
differences of matching genes including disabled genes.

From NEAT paper http://nn.cs.utexas.edu/downloads/papers/stanley.ec02.pdf
"""


def generate_gene_metric(c_1=1.0, c_2=1.0, c_3=0.4):
    def _compare_gene_difference(genes_1, genes_2):
        last_match = [0, 0]
        W, M, i, j = (0, 0, 0, 0)
        while True:
            if genes_1[i].innov == genes_2[j].innov:
                last_match = [i, j]
                W += abs(genes_1[i].weight - genes_2[j].weight)
                M, i, j = (M + 1, i + 1, j + 1)
            elif genes_1[i].innov < genes_2[j].innov:
                i += 1
            elif genes_1[i].innov > genes_2[j].innov:
                j += 1
            if i == len(genes_1) or j == len(genes_2):
                break
        N = max(len(genes_1), len(genes_2))
        D = 2 + last_match[0] + last_match[1] - 2 * M
        E = len(genes_1) + len(genes_2) - last_match[0] - last_match[1] - 2
        W = W / M
        return c_1 * E / N + c_2 * D / N + c_3 * W
    return _compare_gene_difference


def generate_neat_metric(c_1=1.0, c_2=1.0, c_3=0.4):
    """Builds a metric on reduced representation of genomes."""
    gen_metric = generate_gene_metric(c_1, c_2, c_3)

    def metric(genome_1, genome_2):
        d = gen_metric(genome_1.nodes, genome_2.nodes)
        d += gen_metric(genome_1.edges, genome_2.edges)
        return d
    return metric
