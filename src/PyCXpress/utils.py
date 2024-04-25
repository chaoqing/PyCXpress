from typing import Tuple

import logging

import numpy as np
from numpy.typing import DTypeLike

logger = logging.getLogger("PyCXpress")


def get_c_type(t: DTypeLike) -> Tuple[str, int]:
    dtype = np.dtype(t)
    relation = {
        np.dtype("bool"): "bool",
        np.dtype("int8"): "int8_t",
        np.dtype("int16"): "int16_t",
        np.dtype("int32"): "int32_t",
        np.dtype("int64"): "int64_t",
        np.dtype("uint8"): "uint8_t",
        np.dtype("uint16"): "uint16_t",
        np.dtype("uint32"): "uint32_t",
        np.dtype("uint64"): "uint64_t",
        np.dtype("float32"): "float",
        np.dtype("float64"): "double",
    }
    return relation.get(dtype, "char"), dtype.itemsize or 1
