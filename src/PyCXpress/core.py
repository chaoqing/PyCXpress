# mypy: disable_error_code="type-arg,arg-type,union-attr,operator,assignment,misc"
import logging

logger = logging.getLogger(__name__)


from typing import Callable, Dict, Iterable, List, Optional, Tuple, Union

import os
from collections import namedtuple
from dataclasses import dataclass
from enum import Enum, auto

import numpy as np
from numpy.typing import DTypeLike

# import tensorflow as tf


def pycxpress_debugger(
    host: Optional[str] = None,
    port: Optional[int] = None,
    debugger: Optional[str] = None,
):
    if debugger is None:
        return

    if host is None:
        host = os.environ.get("PYCXPRESS_DEBUGGER_HOST", "localhost")

    if port is None:
        port = os.environ.get("PYCXPRESS_DEBUGGER_PORT", 5678)

    if debugger.lower() == "pycharm":
        try:
            import pydevd_pycharm

            pydevd_pycharm.settrace(
                host, port=port, stdoutToServer=True, stderrToServer=True, suspend=True
            )
        except ConnectionRefusedError:
            logger.warning(
                "Can not connect to Python debug server (maybe not started?)"
            )
            logger.warning(
                "Use PYCXPRESS_DEBUGGER_TYPE=debugpy instead as Pycharm professional edition is needed for Python debug server feature."
            )
    elif debugger.lower() == "debugpy":
        import debugpy

        debugpy.listen((host, port))
        logger.info(f"debugpy listen on {host}:{port}, please use VSCode to attach")
        debugpy.wait_for_client()
    else:
        logger.warning(
            f"Only PYCXPRESS_DEBUGGER_TYPE=debugpy|pycharm supported but {debugger} provided"
        )


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


@dataclass
class TensorMeta:
    dtype: DTypeLike  # the data type similar to np.int_
    shape: Union[
        int, Iterable[int], Callable[..., Union[int, Iterable[int]]]
    ]  # the maximal size of each dimension
    name: Optional[str] = None
    doc: Optional[str] = None

    def to_dict(self, *args, **kwargs) -> Dict:
        assert self.name is not None

        max_size = self.shape
        if isinstance(max_size, Callable):
            max_size = max_size(*args, **kwargs)
        max_size = tuple(max_size) if isinstance(self.shape, Iterable) else (max_size,)

        dtype, itemsize = get_c_type(self.dtype)

        return {
            "name": self.name,
            "dtype": dtype,
            "shape": tuple(round(-i) if i < 0 else None for i in max_size),
            "buffer_size": np.prod([round(abs(i)) for i in max_size]) * itemsize,
            "doc": f"" if self.doc is None else self.doc,
        }

    def setdefault(self, name: str) -> str:
        if self.name is None:
            self.name = name
        return self.name


class ModelAnnotationType(Enum):
    Input = auto()
    Output = auto()
    Operator = auto()
    HyperParams = auto()


class ModelRuntimeType(Enum):
    GraphExecution = auto()
    EagerExecution = auto()
    OfflineExecution = auto()


@dataclass
class TensorWithShape:
    data: Optional[np.ndarray] = None
    shape: Optional[Tuple] = None


class ModelAnnotationCreator(type):
    def __new__(
        mcs,
        name,
        bases,
        attrs,
        fields: Dict[str, TensorMeta],
        type: ModelAnnotationType,
        mode: ModelRuntimeType,
    ):
        if type == ModelAnnotationType.Input:
            generate_property = mcs.generate_input_property
        elif type == ModelAnnotationType.Output:
            generate_property = mcs.generate_output_property
        else:
            raise NotImplementedError()

        for field_name, field_meta in fields.items():
            field_meta.setdefault(field_name)
            attrs[field_name] = generate_property(field_meta)

        get_buffer_shape, set_buffer_value, init_func = mcs.general_funcs(
            name, [field_meta.name for field_meta in fields.values()]
        )

        attrs["__init__"] = init_func
        attrs["set_buffer_value"] = set_buffer_value
        if type == ModelAnnotationType.Output:
            attrs["get_buffer_shape"] = get_buffer_shape
        attrs.setdefault("__slots__", []).append("__buffer_data__")

        return super().__new__(mcs, name, bases, attrs)

    @staticmethod
    def general_funcs(name: str, field_names: List[str]):
        def get_buffer_shape(self, name: str):
            buffer = getattr(self.__buffer_data__, name)
            return buffer.shape

        def set_buffer_value(self, name: str, value):
            buffer = getattr(self.__buffer_data__, name)
            buffer.data = value

        def init_func(self):
            _BufferData_ = namedtuple("_BufferData_", field_names)
            self.__buffer_data__ = _BufferData_(
                *tuple(TensorWithShape() for _ in field_names)
            )

        return get_buffer_shape, set_buffer_value, init_func

    @staticmethod
    def generate_input_property(field: TensorMeta):
        def get_func(self):
            return getattr(self.__buffer_data__, field.name).data

        def set_func(*_):
            raise AssertionError("Not supported for input tensor")

        def del_func(_):
            raise AssertionError("Not supported for input tensor")

        return property(fget=get_func, fset=set_func, fdel=del_func, doc=field.doc)

    @staticmethod
    def generate_output_property(field: TensorMeta):
        def get_func(self):
            logger.warning(f"Only read the data field {field.name} in debugging mode")
            buffer = getattr(self.__buffer_data__, field.name)
            return buffer.data[: np.prod(buffer.shape)].reshape(buffer.shape)

        def set_func(self, data):
            buffer = getattr(self.__buffer_data__, field.name)
            buffer.shape = data.shape
            buffer.data[: np.prod(data.shape)] = data.flatten()

        def del_func(_):
            raise AssertionError("Not supported for output tensor")

        return property(fget=get_func, fset=set_func, fdel=del_func, doc=field.doc)


def convert_to_spec_tuple(fields: Iterable[TensorMeta]):
    return tuple(
        (v["name"], v["dtype"], v["buffer_size"]) for v in [v.to_dict() for v in fields]
    )


def main():
    pass


if __name__ == "__main__":
    main()
