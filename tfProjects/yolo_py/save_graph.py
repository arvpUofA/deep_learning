# -*- coding: utf-8 -*-
import tensorflow as tf
import os, argparse
try:
  from tensorflow.contrib import tensorrt as trt
  tensorRT_installed = True
except ImportError as e:
  print e
  print "Are you sure tensorflow is installed with tensorRT?"
  print "Saving graph using model not optimized by tensorRT"
  tensorRT_installed = False

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('model_dir', 'checkpoints/yolov3-tiny', 'Model folder')
tf.app.flags.DEFINE_string('graph_name', 'frozen_graph', 'Graph name')
tf.app.flags.DEFINE_string('graph_dir', 'graphs/yolov3-tiny', 'Graph folder')

tf.app.flags.DEFINE_string('output_node_names', "detection_boxes", "Names of the output nodes, comma separated.")

tf.app.flags.DEFINE_boolean('tensorRT', False, 'Optimize using tensorRT')
tf.app.flags.DEFINE_string('precision', 'FP32', "Precision of the graph FP32 or FP16")
# The original freeze_graph function
# from tensorflow.python.tools.freeze_graph import freeze_graph 
dir = os.path.dirname(os.path.realpath(__file__))

def freeze_graph(model_dir, output_node_names, tensorRT=False):
  """Extract the sub graph defined by the output nodes and convert 
  all its variables into constant 
  Args:
    model_dir: the root folder containing the checkpoint state file
    output_node_names: a string, containing all the output node's names, comma separated
  """
  if not tf.gfile.Exists(model_dir):
    raise AssertionError(
      "Export directory doesn't exists. Please specify an export "
      "directory: %s" % model_dir)

  if not output_node_names:
    print("You need to supply the name of a node to --output_node_names.")
    return -1

  # We retrieve our checkpoint fullpath
  checkpoint = tf.train.get_checkpoint_state(model_dir)
  input_checkpoint = checkpoint.model_checkpoint_path
  
  # We precise the file fullname of our freezed graph
  absolute_model_dir = "/".join(input_checkpoint.split('/')[:-1])
  graph_name = FLAGS.graph_name
  output_graph = absolute_model_dir + "/" + graph_name

  # We clear devices to allow TensorFlow to control on which device it will load operations
  clear_devices = True

  # We start a session using a temporary fresh Graph
  with tf.Session(graph=tf.Graph()) as sess:
    # We import the meta graph in the current default Graph
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=clear_devices)

    # We restore the weights
    saver.restore(sess, input_checkpoint)

    # We use a built-in TF helper to export variables to constants
    output_graph_def = tf.graph_util.convert_variables_to_constants(
      sess, # The session is used to retrieve the weights
      tf.get_default_graph().as_graph_def(), # The graph_def is used to retrieve the nodes 
      output_node_names.split(",") # The output node names are used to select the usefull nodes
    ) 

    output_graph_file = FLAGS.graph_dir + '/' + graph_name + ".pb"
    if tensorRT:
      workspace_size = 1 << 30 # should these be command line args?
      batch_size = 1

      trt_graph = trt.create_inference_graph(
        output_graph_def,
        output_node_names.split(","),
        max_batch_size=batch_size,
        max_workspace_size_bytes=workspace_size,
        precision_mode=FLAGS.precision
      )
      tf.train.write_graph(sess.graph_def, FLAGS.graph_dir, graph_name+'.pb.txt')

      with tf.gfile.GFile(output_graph_file, "wb") as f:
        f.write(trt_graph.SerializeToString())
    else:
      tf.train.write_graph(sess.graph_def, absolute_model_dir, graph_name+'.pb.txt')
      # Finally we serialize and dump the output graph to the filesystem
      with tf.gfile.GFile(output_graph_file, "wb") as f:
        f.write(output_graph_def.SerializeToString())
      print("%d ops in the final graph." % len(output_graph_def.node))

  return output_graph_def

def main(argv=None):
  use_tensorRT = FLAGS.tensorRT and tensorRT_installed
  freeze_graph(FLAGS.model_dir, FLAGS.output_node_names, use_tensorRT)

if __name__ == '__main__':
  tf.app.run()