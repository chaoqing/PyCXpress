import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3' 
import logging
from contextlib import nullcontext

logger = logging.getLogger(__name__)

import numpy as np

import tensorflow as tf

def debug_array(a: np.array):
    logger.info(f"array data type: {a.dtype}")
    logger.info(f"array data shape: {a.shape}")
    logger.info(f"array data: ")
    logger.info(a)


class InputDataSet(object):
    def __init__(self):
        self.data = {}

    def set(self, name:str, x: np.array):
        self.data[name] = x

    @property
    def input_a(self):
        return tf.Variable(self.data['input_a'], name="input_a")

    @property
    def input_b(self):
        return tf.Variable(self.data['input_b'], name="input_b")

class OutputDataSet(object):
    def __init__(self):
        self.data = {}

    def set_buffer(self, name: str, x: np.array):
        assert x.ndim == 1
        self.data[name] = {"data": x}

    def get_shape(self, name: str):
        return self.data[name]['shape']

    @property
    def output_a(self, x: tf.Variable):
        raise NotImplementedError

    @output_a.setter
    def output_a(self, x: tf.Variable):
        buffer = self.data['output_a']
        buffer['shape'] = x.shape
        buffer['data'][:np.prod(x.shape)] = x.numpy().flatten()


def main():
    pass

if __name__ == "__main__":
    main()