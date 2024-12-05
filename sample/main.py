#!/usr/bin//env python

from typing import List

import shutil
import sys
from contextlib import nullcontext

import tensorflow as tf
import tensorflow.compat.v1 as tf1
from tensorflow.python.saved_model import tag_constants


def main(argv: List[str]):
    shutil.rmtree("./frozen_graph", ignore_errors=True)
    shutil.rmtree("./saved_model", ignore_errors=True)

    g = tf.Graph()

    with g.as_default() if True else nullcontext():
        a = tf1.placeholder(tf.uint8, shape=(None, None), name="a")
        b = tf1.placeholder(tf.uint8, shape=(None, None), name="b")
        _ = a + b

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
