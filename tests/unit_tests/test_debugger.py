import unittest
from gerel.debug.class_debug_decorator import add_inst_validator

MSG = 'value greater than ten'


def simple_validator(self, *args, **kwargs):
    if self.value > 10:
        raise Exception(MSG)


@add_inst_validator(env="TESTING", validator=simple_validator)
class Test:
    def __init__(self, value):
        self.value = value

    def set_value(self, value):
        self.value = value


class TestClassValidator(unittest.TestCase):
    def testClassValidator(self):
        t = Test(1)
        t.set_value(5)

        with self.assertRaises(Exception) as context:
            t.set_value(11)

        self.assertEqual(MSG, str(context.exception))

