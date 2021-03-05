import itertools


class Node:
    """Node."""

    innov_iter = itertools.count()
    registry = {}

    def __init__(self, layer_num, layer_ind, weight):
        innov = Node.registry.get((layer_num, layer_ind), None)
        innov = innov if innov is not None else next(Node.innov_iter)
        Node.registry[(layer_num, layer_ind)] = innov
        self.innov = innov
        self.layer_num = layer_num
        self.layer_ind = layer_ind
        self.weight = weight
        self.edges_out = []
        self.edges_in = []


    @classmethod
    def copy(cls, node):
        """Copies the node location and innovation number.

        Note the edges_out and edges_in are not copied. This is taken care of
        in the Edge initialization step itself.
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
        return self.layer_num, self.layer_ind, self.innov, self.weight
