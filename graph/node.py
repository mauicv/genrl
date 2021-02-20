import itertools


class Node:
    """Node."""

    innov_iter = itertools.count()

    def __init__(self, layer_num, layer_ind, innov=None):
        self.layer_num = layer_num
        self.layer_ind = layer_ind
        self.innov = innov if innov is not None else next(Node.innov_iter)
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
            innov=node.innov)
