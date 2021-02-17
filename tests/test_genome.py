import unittest
from graph.genome import Genome

class TestGenomeClass(unittest.TestCase):
    """Test methods assoicated to Genome class."""

    def test_genome_init(self):
        """Test genome __init__ function.

        New instance of genome contains specified number of input and output
        nodes plus a single connecting node in the middle.
        """
        g = Genome(input_size=2, output_size=3, depth=5)

        # Check correct nodes
        self.assertEqual(len(g.inputs), 2)
        self.assertEqual(len(g.nodes), 6)
        self.assertEqual(len(g.outputs), 3)
        for node in g.inputs:
            self.assertEqual(node.layer_num, 0)
        for node in g.outputs:
            self.assertEqual(node.layer_num, len(g.layers) -1)
        self.assertEqual(len(g.layers), 7)

        # check central connecting node edges
        self.assertEqual(
            g.layer_edges_in(layer_num=6),
            g.layer_edges_out(layer_num=1)
        )
        self.assertEqual(
            g.layer_edges_in(layer_num=1),
            g.layer_edges_out(layer_num=0)
        )

    def test_get_addmissable_edges(self):
        g = Genome(input_size=2, output_size=3, depth=5)
        n2 = g.add_node(3)
        g.add_edge(g.layers[0][0], n2)
        g.add_edge(n2, g.outputs[0])

        self.assertEqual(len(g.layer_edges_in(3)), 1)
        self.assertEqual(len(g.layer_edges_out(3)), 1)

        for layer_num in [2,4,5]:
            self.assertEqual(len(g.layer_edges_in(layer_num)), 0)
            self.assertEqual(len(g.layer_edges_out(layer_num)), 0)

        self.assertEqual(len(g.get_addmissable_edges()), 5)
        addmissable = lambda e: e.to_node.layer_num - e.from_node.layer_num > 1
        for e in g.get_addmissable_edges():
            self.assertEqual(addmissable(e), True)
