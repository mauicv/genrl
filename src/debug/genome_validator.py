def validate_genome(genome):
    edge_innov = set(e.innov for e in genome.edges)
    node_innov = set(n.innov for n in genome.nodes)
    truth = len(edge_innov) == len(genome.edges)
    truth = truth and len(node_innov) == len(genome.nodes)
    for e in genome.edges:
        truth = truth and (e.from_node.innov, e.to_node.innov) in genome.edge_innovs
    for e_1, e_2 in zip(genome.edges, genome.edges[1:]):
        truth = truth and e_1.innov < e_2.innov
    return truth
