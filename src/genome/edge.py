import itertools


class Edge:
    """Edge class.

    Represents a directed edge on a genome (graph structure). Stores
    reference to from_node and to_node. A weight of the edge and
    also it's innovation number.
    """

    innov_iter = itertools.count()
    registry = {}

    def __init__(self, from_node, to_node, weight, active=True):
        """Creates edge between from_node and to_node with
        specified weight and activity.

        :param from_node: Source Node in genome
        :param to_node: Target Node in genome
        :param weight: Weight of edge, represents contribution of
            from_nodes input to to_node
        :param active: If edge not active it will not effect a
            model class built with to_reduced_repr object.

        :raises ValueError: to_node must be a member of a layer
            further on in the genome.
        """
        innov = Edge.registry.get((from_node.innov, to_node.innov), None)
        innov = innov if innov is not None else next(Edge.innov_iter)
        Edge.registry[(from_node.innov, to_node.innov)] = innov
        self.innov = innov
        if to_node.layer_num - from_node.layer_num < 1:
            raise ValueError('Cannot connect edge to lower or same layer')
        self.active = active
        self.from_node = from_node
        self.to_node = to_node
        self.weight = weight
        from_node.edges_out.append(self)
        to_node.edges_in.append(self)

    @classmethod
    def copy(cls,
             edge,
             new_genome):
        """Copies a edge onto a genome.

        :param edge: Edge to copy and place on new_genome
        :param new_genome: Target of copied edge.
        :return: The copied edge.
        :raises ValueError: from and to nodes must exist in the target
            genome. The added edges innovation number must be in order
            with current edge innovation numbers.
        """
        fn = edge.from_node
        tn = edge.to_node

        if len(new_genome.layers) - 1 < fn.layer_num or \
                len(new_genome.layers[fn.layer_num]) - 1 < fn.layer_ind:
            raise ValueError('from_node does not exist on new_genome.')

        if len(new_genome.layers) - 1 < tn.layer_num or \
                len(new_genome.layers[tn.layer_num]) - 1 < tn.layer_ind:
            raise ValueError('to_node does not exist on new_genome.')

        from_node = new_genome.layers[fn.layer_num][fn.layer_ind]
        to_node = new_genome.layers[tn.layer_num][tn.layer_ind]

        edge = cls(from_node, to_node, edge.weight)
        if new_genome.edges and new_genome.edges[-1].innov > edge.innov:
            raise ValueError('innovation numbers are out of order.')
        new_genome.edges.append(edge)
        return edge

    @property
    def to_reduced_repr(self):
        """Creates simple serializable representation.

        :return: tuple of index and property values.
        """
        return (
            self.from_node.to_reduced_repr,
            self.to_node.to_reduced_repr,
            self.weight,
            self.innov,
            self.active
        )
