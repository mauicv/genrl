def validate_genome(genome, *args, **kwargs):
    if len(set(e.innov for e in genome.edges)) != len(genome.edges):
        raise ValueError('Non-unique edge in genome edges.')

    if len(set(n.innov for n in genome.nodes)) != len(genome.nodes):
        raise ValueError('Non-unique node in genome node.')

    for e in genome.edges:
        if not (e.from_node.innov, e.to_node.innov) in genome.edge_innovs:
            raise ValueError('Edge innovation not registered.')

    for e_1, e_2 in zip(genome.edges, genome.edges[1:]):
        if not e_1.innov < e_2.innov:
            raise ValueError('Edge innovations are out of order.')

    for n_1, n_2 in zip(genome.nodes, genome.nodes[1:]):
        if not n_1.innov < n_2.innov:
            raise ValueError('Node innovations are out of order.')

    for n in genome.inputs:
        if not n.type == 'input':
            raise ValueError('Non input node in inputs.')

    for n in genome.outputs:
        if not n.type == 'output':
            raise ValueError('Non output node in outputs.')
