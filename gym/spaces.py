from dataclasses import dataclass
import numpy as np


@dataclass
class Box:
    shape: int
    high: list
    low: list

    def sample(self):
        return np.clip(np.random.normal(0, 0.01, size=(self.shape)), -1, 1) \
            * (self.high - self.low)

    @property
    def arc_sizes(self):
        return [high - low for low, high in zip(self.low, self.high)]
