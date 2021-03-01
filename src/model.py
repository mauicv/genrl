"""Model.py

Transforms Genome object into a graph structure.

TODO:
- Implement tf_model_from_gene https://github.com/crisbodnar/TensorFlow-NEAT/blob/master/tf_neat/recurrent_net.py
- Implement biases and activation functions.
"""
import numpy as np
from src.activations import step


class Model:
    def __init__(self, genes):
        nodes, edges = genes
        num_layer = max([layer for layer, _, _, _ in nodes])
        self.cells = {}
        self.layers = []
        for from_layer in range(num_layer):
            from_nodes = [(node_layer, node_ind, bias) for node_layer, node_ind, _, bias
                          in nodes if from_layer == node_layer]
            from_edges = [(from_node_layer, from_node_layer_ind, to_node_layer, to_node_layer_ind, weight) for
                          (from_node_layer, from_node_layer_ind, _, _), (to_node_layer, to_node_layer_ind, _, _),
                          weight, _ in edges if from_node_layer == from_layer]
            to_nodes = list(set((to_node_layer, to_node_layer_ind, bias) for
                        (from_node_layer, _, _, _), (to_node_layer, to_node_layer_ind, _, bias),
                        weight, _ in edges if from_node_layer == from_layer))

            for i, j, b in [*from_nodes, *to_nodes]:
                self.add_cell(i, j, b)

            dims = (len(from_nodes), len(to_nodes))
            layer = Layer(dims, self.cells, from_edges)
            self.layers.append(layer)

        self.constants = [self.cells[(i, j)] for i, j, innov, _ in nodes if innov == -1]
        self.inputs = self.layers[0].inputs
        self.outputs = [cell for cell in self.constants if cell.i != 0]

    def add_cell(self, i, j, b):
        cell = self.cells.get((i, j), None)
        if not cell:
            cell = Cell(i, j, b)
            self.cells[(i, j)] = cell
        return cell

    def __call__(self, inputs):
        for cell, val in zip(self.inputs, inputs):
            cell.acc = val
        self.layers[0].run(activation=lambda x:x)
        for layer in self.layers[1:]:
            layer.run()
        return [cell.acc + cell.b for cell in self.outputs]


class Layer:
    def __init__(self, dims, cells, edges):
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

    def run(self, activation=step):
        input_vals = np.array([activation(cell.acc + cell.b) for cell in self.inputs])
        output_vals = self.mat.T @ input_vals
        for val, cell in zip(output_vals, self.outputs):
            cell.acc += val

class Cell:
    def __init__(self, i, j, b):
        self.i = i
        self.j = j
        self.b = b
        self.acc = 0