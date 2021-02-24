import unittest
from graph.util import sample_weight
from unittest.mock import Mock
import random


class TestUtilMethods(unittest.TestCase):
    """Test methods assoicated to Genome class."""

    def test_sample_weight(self):
        random.random = Mock(return_value=0.1)
        self.assertEqual(sample_weight(-2, 2), random.random() * 4 - 2)
