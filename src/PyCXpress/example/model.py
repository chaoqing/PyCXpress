# mypy: disable_error_code="arg-type,type-arg,attr-defined"
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import logging

logging.basicConfig(level=logging.DEBUG)

import os
from contextlib import nullcontext

import numpy as np

from PyCXpress import (
    ModelAnnotationCreator,
    ModelAnnotationType,
    ModelRuntimeType,
    TensorMeta,
    convert_to_spec_tuple,
    pycxpress_debugger,
)


def show(a: np.ndarray):
    logging.info(f"array data type: {a.dtype}")
    logging.info(f"array data shape: {a.shape}")
    logging.info(f"array data: ")
    logging.info(a)


InputFields = dict(
    data_to_be_reshaped=TensorMeta(
        name="input/data",
        dtype=np.float_,
        shape=(100,),
    ),
    new_2d_shape=TensorMeta(
        dtype=np.uint8,
        shape=-2,
    ),
)


class InputDataSet(
    metaclass=ModelAnnotationCreator,
    fields=InputFields,
    type=ModelAnnotationType.Input,
    mode=ModelRuntimeType.EagerExecution,
    raw=False,
):
    pass


OutputFields = dict(
    output_a=TensorMeta(
        dtype=np.float_,
        shape=(10, 10),
    ),
)


class OutputDataSet(
    metaclass=ModelAnnotationCreator,
    fields=OutputFields,
    type=ModelAnnotationType.Output,
    mode=ModelRuntimeType.EagerExecution,
    raw=False,
):
    pass


class Model:
    def __init__(self):
        self.input = None
        self.output = None

    def initialize(self):
        self.input, self.output = InputDataSet(), OutputDataSet()

        return (
            self.input,
            self.output,
            tuple(convert_to_spec_tuple(InputFields.values(), OutputFields.values())),
        )

    def run(self):
        self.model(self.input, self.output)

    @staticmethod
    def model(input: InputDataSet, output: OutputDataSet, use_tensorflow: bool = True):
        with nullcontext():
            # print(input.data_to_be_reshaped)
            # print(input.new_2d_shape)
            if use_tensorflow:
                import tensorflow as tf

                output.output_a = tf.transpose(
                    tf.reshape(
                        input.data_to_be_reshaped, tf.cast(input.new_2d_shape, tf.int32)
                    )
                )
            else:
                output.output_a = input.data_to_be_reshaped.reshape(
                    input.new_2d_shape
                ).T
            # print(output.output_a)


def main():

    model = Model()
    input_data, output_data, spec = model.initialize()
    print(spec)

    input_data.set_buffer_value("input/data", np.arange(12, dtype=np.float_))
    print(input_data.data_to_be_reshaped)
    input_data.set_buffer_value("new_2d_shape", np.array([3, 4]).astype(np.uint8))
    print(input_data.new_2d_shape)
    output_data.set_buffer_value("output_a", np.arange(12) * 0)

    model.run()
    print(output_data.output_a)
    print(output_data.get_buffer_shape("output_a"))

    # pycxpress_debugger()


if __name__ == "__main__":
    main()
