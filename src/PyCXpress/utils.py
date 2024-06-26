from typing import Any, Dict, Tuple

import logging
import os

import numpy as np
from numpy.typing import DTypeLike

logger = logging.getLogger("PyCXpress")


# mypy: disable-error-code="attr-defined"
def _refresh_environ():
    """https://discuss.python.org/t/method-to-refresh-os-environ/54774/11"""
    import ctypes
    import sys

    if sys.platform == "linux":

        def _get_env_array(*, lib=ctypes.CDLL(None)):
            return ctypes.POINTER(ctypes.c_char_p).in_dll(lib, "environ")

    elif sys.platform == "win32":

        def _get_env_array(*, lib=ctypes.CDLL("ucrtbase")):
            p = ctypes.CFUNCTYPE(ctypes.POINTER(ctypes.POINTER(ctypes.c_wchar_p)))
            return p(("__p__wenviron", lib))()[0]

    else:
        raise ImportError

    uppercase_names = sys.platform == "win32"
    _env_array = _get_env_array()
    if isinstance(_env_array[0], bytes):
        equals = b"="
    else:
        equals = "="  # type: ignore
    c_environ = {}
    for entry in _env_array:
        if entry is None:
            break
        name, value = entry.split(equals, 1)
        if uppercase_names:
            c_environ[name.upper()] = value
        else:
            c_environ[name] = value

    os.environ._data.clear()
    os.environ._data.update(c_environ)


def getenv(key, default=None):
    """Just like getenv but always do refresh to obtain the latest process environments"""
    _refresh_environ()
    return os.getenv(key, default)


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


class Singleton(type):
    _instances: Dict[Any, Any] = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls._instances:
            cls._instances[cls] = super().__call__(*args, **kwargs)
        return cls._instances[cls]
