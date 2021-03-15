from src.genome.genome import Genome
from src.genome.edge import Edge
from src.genome.node import Node
import itertools
import numpy as np


def default_gen():
    while True:
        yield 1


def genome_pair_factory(
        weight_gen=default_gen(),
        bias_gen=default_gen()
    ):
    Node.innov_iter = itertools.count()
    Edge.innov_iter = itertools.count()

    np.random.seed(1)

    g1 = Genome.default(input_size=2, output_size=3, depth=5)
    n1 = g1.add_node(4)
    g1.add_edge(g1.layers[0][0], n1)
    g1.add_edge(n1, g1.outputs[0])

    n2 = g1.add_node(3)
    g1.add_edge(g1.layers[0][1], n2)
    g1.add_edge(n2, g1.outputs[2])

    g2 = Genome.copy(g1)

    n4 = g2.add_node(2)
    g2.add_edge(g2.layers[0][1], n4)
    g2.add_edge(n4, g2.layers[3][0])

    n5 = g1.add_node(4)
    g1.add_edge(g1.layers[3][0], n5)
    g1.add_edge(n5, g1.outputs[0])

    n3 = g1.add_node(3)
    e1 = g1.add_edge(g1.layers[0][0], n3)
    e2 = g1.add_edge(n3, g1.outputs[2])
    g2.layers[3].append(Node.copy(n3))
    g2.nodes.append(Node.copy(n3))

    n6 = g1.add_node(3)
    g1.add_edge(g1.layers[0][0], n6)
    g1.add_edge(n6, g1.outputs[1])

    g2.add_edge(g2.layers[0][0], n3)
    e2 = g2.add_edge(n3, g2.outputs[2])

    for w, edge in zip(weight_gen, [*g1.edges, *g2.edges]):
        edge.weight = w
    for w, node in zip(bias_gen, [*g1.nodes, *g2.nodes]):
        node.weight = w

    return g1, g2


def genome_factory(
        weight_gen=default_gen(),
        bias_gen=default_gen()
    ):
    Node.innov_iter = itertools.count()
    Edge.innov_iter = itertools.count()

    np.random.seed(1)

    g1 = Genome.default(input_size=2, output_size=3, depth=5)
    n1 = g1.add_node(4)
    g1.add_edge(g1.layers[0][0], n1)
    g1.add_edge(n1, g1.outputs[0])

    n2 = g1.add_node(3)
    g1.add_edge(g1.layers[0][1], n2)
    g1.add_edge(n2, g1.outputs[2])
    n5 = g1.add_node(4)
    g1.add_edge(g1.layers[3][0], n5)
    g1.add_edge(n5, g1.outputs[0])

    n3 = g1.add_node(3)
    e1 = g1.add_edge(g1.layers[0][0], n3)
    e2 = g1.add_edge(n3, g1.outputs[2])
    n6 = g1.add_node(3)
    g1.add_edge(g1.layers[0][0], n6)
    g1.add_edge(n6, g1.outputs[1])

    for w, edge in zip(weight_gen, g1.edges):
        edge.weight = w

    for w, node in zip(bias_gen, g1.nodes):
        node.weight = w

    return g1
