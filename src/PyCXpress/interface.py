# mypy: disable_error_code="type-arg"
from typing import List, Protocol, Tuple

from abc import abstractmethod

import numpy as np


class InputTensorProtocol(Protocol):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def set_buffer_value(self, name: str, value: np.ndarray) -> None: ...


class OutputTensorProtocol(Protocol):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def get_buffer_shape(self, name: str) -> Tuple[int]: ...

    @abstractmethod
    def set_buffer_value(self, name: str, value: np.ndarray) -> None: ...


TensorBufferProtocol = Tuple[
    str,  # name
    str,  # dtype
    int,  # buffer size
    bool,  # is_output
]


class ModelProtocol(Protocol):
    def __init__(self, *args, **kwargs): ...

    @abstractmethod
    def initialize(
        self,
    ) -> Tuple[
        InputTensorProtocol, OutputTensorProtocol, Tuple[TensorBufferProtocol]
    ]: ...

    @abstractmethod
    def run(self) -> None: ...

    @abstractmethod
    def reset(self) -> None: ...
