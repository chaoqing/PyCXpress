"""PyCXpress is a high-performance hybrid framework that seamlessly integrates Python and C++ to harness the flexibility of Python and the speed of C++ for efficient and expressive computation, particularly in the realm of deep learning and numerical computing."""

__all__ = [
    "TensorMeta",
    "ModelAnnotationCreator",
    "ModelAnnotationType",
    "ModelRuntimeType",
    "convert_to_spec_tuple",
    "pycxpress_debugger",
    "get_include",
    "version",
]

from importlib import metadata as importlib_metadata
from pathlib import Path


def get_version() -> str:
    try:
        return importlib_metadata.version(__name__)
    except importlib_metadata.PackageNotFoundError:  # pragma: no cover
        return "unknown"


version: str = get_version()


from .core import (
    ModelAnnotationCreator,
    ModelAnnotationType,
    ModelRuntimeType,
    TensorMeta,
    convert_to_spec_tuple,
)
from .debugger import pycxpress_debugger


def get_include() -> str:
    return str(Path(__file__).parent.absolute() / "include")
