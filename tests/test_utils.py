import unittest
from src.util import sample_weight
from unittest.mock import patch
import random


class TestUtilMethods(unittest.TestCase):
    """Test methods assoicated to Genome class."""

    def test_sample_weight(self):
        with patch('random.random', side_effect=[0.1, 0.1]):
            self.assertEqual(sample_weight(-2, 2), random.random() * 4 - 2)
