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


def add_two_numpy_with_tf(a: np.array, b: np.array, res: np.array):
    with nullcontext():
          c = (tf.Variable(a, name="a")+tf.Variable(b, name="b")).numpy()
          res[:c.size] = c.flatten()
          print(c)
          return tuple(c.shape)

def main():
    pass

if __name__ == "__main__":
    main()