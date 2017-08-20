#!/usr/bin/env python
# coding: utf-8

"""
__doc__
General code to convert a trained keras model into an inference tensorflow model.
"""

__author__ = "Haruyuki Ichino"
__version__ = "1.0"
__date__ = "2017/08/20"

print(__doc__)


import sys

from keras.models import load_model
import tensorflow as tf
import os
import os.path as osp
from keras import backend as K

from tensorflow.python.framework import graph_util
from tensorflow.python.framework import graph_io


# Set parameters
if len(sys.argv) != 2:
    print("Usage: python keras_to_tensorflow.py [keras model file path]")
    sys.exit(1)
keras_model_path = sys.argv[1]
num_output = 1
write_graph_def_ascii_flag = True
prefix_output_node_names_of_final_network = 'output_node'
keras_model_name = keras_model_path.split("/")[-1].split(".")[0]
tensorflow_graph_name = keras_model_name + '.pb'

# 出力ディレクトリの準備
output_dir = './tensorflow_model/'
if not os.path.isdir(output_dir):
    os.mkdir(output_dir)


# Load keras model and rename output
K.set_learning_phase(0)
keras_model = load_model(keras_model_path)

pred = [None]*num_output
pred_node_names = [None]*num_output
for i in range(num_output):
    pred_node_names[i] = prefix_output_node_names_of_final_network+str(i)
    pred[i] = tf.identity(keras_model.output[i], name=pred_node_names[i])
print('Output nodes names: ', pred_node_names)


# [optional] write graph definition in ascii
sess = K.get_session()
if write_graph_def_ascii_flag:
    f = tensorflow_graph_name + '.ascii'
    tf.train.write_graph(sess.graph.as_graph_def(), output_dir, f, as_text=True)
    print('Saved the graph definition: ', osp.join(output_dir, f))


# convert variables to constants and save
constant_graph = graph_util.convert_variables_to_constants(sess, sess.graph.as_graph_def(), pred_node_names)
graph_io.write_graph(constant_graph, output_dir, tensorflow_graph_name, as_text=False)

print('Saved the TensorFlow graph: ', osp.join(output_dir, tensorflow_graph_name))
