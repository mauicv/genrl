import unittest
from gerel.util.math import vector_decomp
import math


class TestUtilMethods(unittest.TestCase):
    def test_vector_decomp(self):
        o = [1, 1, 0]
        v = [0, 0, 1]
        p = [2, 0, 1]
        a, b = vector_decomp(p, o, v)
        self.assertEqual(b, math.sqrt(2))
        self.assertEqual(a, 1)