#!/usr/bin//env python

import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
from typing import List, Optional

import logging
import shutil
import sys
from contextlib import nullcontext

logging.basicConfig(level=logging.DEBUG)
import numpy as np
import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.saved_model import tag_constants

from PyCXpress import (
    ModelAnnotationCreator,
    ModelAnnotationType,
    ModelRuntimeType,
    TensorMeta,
    convert_to_spec_tuple,
    pycxpress_debugger,
)
from PyCXpress.utils import getenv

execution_mode = (
    ModelRuntimeType.GraphExecution
    if __name__ == "__main__"
    else ModelRuntimeType.EagerExecution
)

InputFields = dict(
    placeHolder_a=TensorMeta(
        name="a",
        dtype=np.uint8,
        shape=(100, 100),
    ),
    placeHolder_b=TensorMeta(name="b", dtype=np.uint8, shape=(100, 100)),
)


class InputDataSet(
    metaclass=ModelAnnotationCreator,
    fields=InputFields,
    type=ModelAnnotationType.Input,
    mode=execution_mode,
    raw=False,
):
    pass


OutputFields = dict(
    sum=TensorMeta(
        name="import/add:0",
        dtype=np.uint8,
        shape=(100, 100),
    ),
)


class OutputDataSet(
    metaclass=ModelAnnotationCreator,
    fields=OutputFields,
    type=ModelAnnotationType.Output,
    mode=execution_mode,
    raw=False,
):
    pass


class Model:
    def __init__(self):
        self.input, self.output = InputDataSet(), OutputDataSet()

    def initialize(self):
        print("current status: ", getenv("PYCXPRESS_STATUS", ""))

        return (
            self.input,
            self.output,
            tuple(convert_to_spec_tuple(InputFields.values(), OutputFields.values())),
        )

    def run(self):
        print("current status: ", getenv("PYCXPRESS_STATUS", ""))
        self.model(self.input, self.output, use_tensorflow=True)

    @staticmethod
    def model(input: InputDataSet, output: OutputDataSet, use_tensorflow: bool = True):
        with nullcontext():
            output.sum = input.placeHolder_a + input.placeHolder_b


def main(_: List[str]):
    shutil.rmtree("./frozen_graph", ignore_errors=True)
    shutil.rmtree("./saved_model", ignore_errors=True)

    g = tf.Graph()

    with g.as_default() if True else nullcontext():
        model = Model()
        model.initialize()
        model.run()

        tf.io.write_graph(g, "./frozen_graph/", name="frozen_graph.pb", as_text=False)

    with tf.io.gfile.GFile("./frozen_graph/frozen_graph.pb", "rb") as f:
        graph_def = tf1.GraphDef()
        graph_def.ParseFromString(f.read())

    with tf1.Session(graph=tf.Graph()) as sess:
        tf.import_graph_def(graph_def)

        signature = tf1.saved_model.predict_signature_def(
            inputs={
                "a": sess.graph.get_tensor_by_name("import/a:0"),
                "b": sess.graph.get_tensor_by_name("import/b:0"),
            },
            outputs={
                "c": sess.graph.get_tensor_by_name("import/add:0"),
            },
        )

        builder = tf1.saved_model.Builder("./saved_model/")
        builder.add_meta_graph_and_variables(
            sess=sess,
            tags=[tag_constants.SERVING],
            signature_def_map={"serving_default": signature},
        )
        model_name = builder.save().decode()
        print(model_name)


if __name__ == "__main__":
    main(sys.argv[1:])
