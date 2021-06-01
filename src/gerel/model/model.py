"""Model.py

Transforms Genome object into a graph structure that can
run computation.
"""
import numpy as np
from gerel.util.activations import step
from gerel.debug.class_debug_decorator import add_inst_validator
from gerel.debug.model_validator import validate_model


@add_inst_validator(env='TESTING', validator=validate_model)
class Model:
    """Model class that acts as the expression of a genome.

    Genomes by themselves are inert but can be transformed
    into models using Model(genome.to_reduced_repr). We use
    this pattern becuase genome.to_reduced_repr is
    serializable and so can be easily passed to different
    processes or across networks.
    """

    def __init__(self, genes, activation=step, layer_activations=None):
        """ Build Model Instance.

        :param genes: Reduced representation of a genome.
            Looks like:
            (
                [(layer_num, layer_ind, innov, weight, type), ...],
                [(
                    from_node.to_reduced_repr,
                    to_node.to_reduced_repr,
                    weight,
                    innov,
                    active
                ), ...]
            )
        """
        nodes, edges = genes
        num_layer = max([layer for layer, _, _, _, _ in nodes])
        self.cells = {}
        self.layers = []

        if layer_activations and len(layer_activations) != num_layer:
            raise ValueError('Mismatch in number of activation functions and layers')

        if not layer_activations:
            # first activation function defaults to identity mapping
            layer_activations = [lambda x: x, *[activation for layer in range(num_layer - 1)]]

        for from_layer in range(num_layer):
            from_nodes = [(node_layer, node_ind, bias) for node_layer, node_ind, _, bias, _
                          in nodes if from_layer == node_layer]
            activate_weight = lambda w, a: w if a else 0
            from_edges = [(from_node_layer,
                           from_node_layer_ind,
                           to_node_layer,
                           to_node_layer_ind,
                           activate_weight(weight, active))
                for (from_node_layer, from_node_layer_ind, _, _, _),
                    (to_node_layer, to_node_layer_ind, _, _, _),
                    weight, _, active
                in edges if from_node_layer == from_layer]

            to_nodes = list(set((to_node_layer, to_node_layer_ind, bias) for
                                (from_node_layer, _, _, _, _),
                                (to_node_layer, to_node_layer_ind, _, bias, _),
                                weight, _, active in edges if from_node_layer == from_layer))

            for i, j, b in [*from_nodes, *to_nodes]:
                cell = self.cells.get((i, j), None)
                if not cell:
                    cell = Cell(i, j, b)
                    self.cells[(i, j)] = cell

            dims = (len(from_nodes), len(to_nodes))
            layer = Layer(
                dims,
                self.cells,
                from_edges,
                activation=layer_activations[from_layer])
            self.layers.append(layer)

        self.inputs = [self.cells[(i, j)] for i, j, _, _, type in nodes if type == 'input']
        self.outputs = [self.cells[(i, j)] for i, j, _, _, type in nodes if type == 'output']

    def __call__(self, inputs):
        """ Computes model output given input.

        Applies matrix multiplication to each layer states, offsets
        the next layer inputs by there node biases and then activates
        and the nodes according to the activation function:

            1/(1+exp(-c*x))

        :param inputs: list of float values.
        :return: list of float values
        """
        for cell, val in zip(self.inputs, inputs):
            cell.acc = val

        for layer in self.layers:
            layer.run()

        output = [cell.acc + cell.b for cell in self.outputs]
        self.reset()
        return output

    def reset(self):
        """Clears accumulated values after network call.

        :return: None
        """
        for _, cell in self.cells.items():
            cell.acc = 0


class Layer:
    """Layer class.

    Models are made up of layers are parts of the overall Model
    function. They first take input and compute activate signals
    for each node (Here called cell) and then perform the matrix
    multiplication operation associated to the edge weights.
    Thus the output of the layer is the next layers input.

    Note that edge connections aren't nesseserily just between
    adjacent layers in the model. An edge might skip layers so
    when the layer performs its computation it changes all cells
    connected to this layer by an edge.
    """
    def __init__(self, dims, cells, edges, activation=step):
        """ Build Layer object.

        :param dims: length 2 tuple giveing the number of layer
            inputs and outputs.
        :param cells: all cells derived from Nodes that lie in this
            genome layer.
        :param edges: all edges connected with a from_node in this
            layer.
        """
        self.activation = activation
        self.edges = edges
        self.mat = np.zeros(dims)
        input_dim, output_dim = dims
        self.inputs = []
        self.outputs = []
        input_cells_map = {}
        output_cells_map = {}
        for fn_i, fn_j, tn_i, tn_j, w in edges:
            input_cells_map[(fn_i, fn_j)] = input_cells_map.get((fn_i, fn_j), len(self.inputs))
            output_cells_map[(tn_i, tn_j)] = output_cells_map.get((tn_i, tn_j), len(self.outputs))
            if len(self.inputs) == input_cells_map[(fn_i, fn_j)]:
                self.inputs.append(cells[(fn_i, fn_j)])
            if len(self.outputs) == output_cells_map[(tn_i, tn_j)]:
                self.outputs.append(cells[(tn_i, tn_j)])
            self.mat[input_cells_map[(fn_i, fn_j)], output_cells_map[(tn_i, tn_j)]] = w

    def run(self):
        """Performs layer computation and updates relevent Model cells.

        :param activation: function taking float and returning float.
        :return: None
        """
        input_vals = np.array([self.activation(cell.acc + cell.b) for cell in self.inputs])
        output_vals = self.mat.T @ input_vals
        for val, cell in zip(output_vals, self.outputs):
            cell.acc += val


class Cell:
    """Cells Object.

    Used only to store accumulated values of layer outputs as each layer
    is run.
    """
    def __init__(self, i, j, b):
        """ Build Cell object.

        :param i: layer_num
        :param j: layer_ind
        :param b: bias weight
        """
        self.i = i
        self.j = j
        self.b = b
        self.acc = 0
