import itertools


class Edge:
    """Edge."""

    innov_iter = itertools.count()

    def __init__(self, from_node, to_node, weight, innov=None):
        if to_node.layer_num - from_node.layer_num < 1:
            raise ValueError('Cannot connect edge to lower or same layer')
        self.disabled = False
        self.innov = innov if innov is not None else next(Edge.innov_iter)
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        from_node.edges_out.append(self)
        to_node.edges_in.append(self)

    @classmethod
    def copy(cls,
             edge,
             new_genome):
        fn = edge.from_node
        tn = edge.to_node

        if len(new_genome.layers) - 1 < fn.layer_num or \
                len(new_genome.layers[fn.layer_num]) - 1 < fn.layer_ind:
            raise ValueError('from_node does not exist on new_genome.')

        if len(new_genome.layers) - 1 < tn.layer_ind or \
                len(new_genome.layers[tn.layer_num]) - 1 < tn.layer_ind:
            raise ValueError('to_node does not exist on new_genome.')

        from_node = new_genome.layers[fn.layer_num][fn.layer_ind]
        to_node = new_genome.layers[tn.layer_num][tn.layer_ind]
        return cls(from_node, to_node, edge.weight, innov=edge.innov)
