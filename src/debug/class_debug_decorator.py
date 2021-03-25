"""Functions that can be used as decorators to check the state of class datastructures are sound.

The class_validator function adds a validation step to each method in the decorated class if the environment is
in TESTING mode. This is used primarily for testing in order to catch cases where class instances enter states
they shouldn't.

Example:

```
def simple_validator(self, *args, **kwargs):
    if self.value > 10:
        raise ValueError()


@add_inst_validator(env="TESTING", validator=simple_validator)
class Test:
    def __init__(self, value):
        self.value = value

    def set_value(self, value):
        self.value = value
```

"""
import functools


def print_all(cls_inst, *args, **kwargs):
    print('instance:', cls_inst)
    print('args:', *args)
    print('kwargs:', **kwargs)


def decorator(func, validator):
    @functools.wraps(func)
    def new_method(*args, **kwargs):
        ret_val = func(*args, **kwargs)
        validator(*args, **kwargs)
        return ret_val
    return new_method


def add_inst_validator(env='TESTING', validator=print_all):
    def class_decorator(cls):
        if env == 'TESTING':
            for attr, val in cls.__dict__.items():
                if callable(val) and not attr.startswith("__"):
                    setattr(cls, attr, decorator(val, validator))
        return cls
    return class_decorator
