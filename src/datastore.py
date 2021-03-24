import json
import numpy as np
import os
from datetime import datetime


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(NpEncoder, self).default(obj)


class DataStore:
    def __init__(
            self,
            dirname=os.getcwd(),
            name='data',
            params=None):

        params = params if params else {}
        self.params = {
            'datetime': str(datetime.now()),
            **params
        }
        self.name = name
        self.dirname = os.path.join(dirname, self.name)
        if not os.path.exists(self.dirname):
            os.makedirs(self.dirname)

    def save(self, data):
        fname = os.path.join(self.dirname, str(len(os.listdir(self.dirname)) + 1))
        with open(fname, 'w') as file:
            file.write(json.dumps(data, cls=NpEncoder))

    def load(self, index):
        fname = os.path.join(self.dirname, str(index))
        with open(fname, 'r') as file:
            return json.loads(file.read())

    def generations(self):
        for file_name in sorted(os.listdir(self.dirname), key=lambda fn_str: int(fn_str)):
            file_name = os.path.join(self.dirname, file_name)
            with open(file_name, 'r') as file:
                yield json.loads(file.read())
