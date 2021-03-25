import itertools


class Node:
    """Node Object.

    Store weight used as bias. Holds indecies of it's location in
    a genome.
    """

    innov_iter = itertools.count()
    registry = {}

    def __init__(self, layer_num, layer_ind, weight, type='hidden'):
        """Create new Node.

        :param layer_num: genome layer.
        :param layer_ind: Location in genome layer
        :param weight: Bias weight.
        :param type: 'input', 'output' or 'hidden'
        """
        innov = Node.registry.get((layer_num, layer_ind), None)
        innov = innov if innov is not None else next(Node.innov_iter)
        Node.registry[(layer_num, layer_ind)] = innov
        self.innov = innov
        self.layer_num = layer_num
        self.layer_ind = layer_ind
        self.weight = weight
        self.edges_out = []
        self.edges_in = []
        self.type = type


    @classmethod
    def copy(cls, node):
        """Deep copy of Node.

        Note the edges_out and edges_in are not copied. This is taken care of
        in the Edge initialization step itself.

        :param node: Node to be copied.
        :return: Copied Node.
        """
        return cls(
            node.layer_num,
            node.layer_ind,
            node.weight)

    @property
    def bias(self):
        return self.weight

    @property
    def to_reduced_repr(self):
        """Returns compact serializable data structure representing Node.

        :return: Tuple like: (layer_num, layer_ind, innov, weight, type)
        """
        return self.layer_num, self.layer_ind, self.innov, self.weight, self.type
