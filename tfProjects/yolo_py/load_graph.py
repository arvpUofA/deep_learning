# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import time

from utils import yoloUtils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_names', './names/dice.names', 'File with class names')
tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.25, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')

tf.app.flags.DEFINE_string('graph_dir', 'graphs/yolov3-tiny', 'Graph folder')
tf.app.flags.DEFINE_string('graph_name', 'frozen_graph', 'Graph name')

tf.app.flags.DEFINE_string('video_file_name', "/home/jkc1/arvp/videos/front-cam-dice-roulette.mp4", "Video to run")

def load_graph(frozen_graph_filename):
  # We load the protobuf file from the disk and parse it to retrieve the 
  # unserialized graph_def
  with tf.gfile.GFile(frozen_graph_filename, "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

  # Then, we import the graph_def into a new Graph and return it 
  with tf.Graph().as_default() as graph:
    # The name var will prefix every op/nodes in your graph
    # Since we load everything in a new graph, this is not needed
    tf.import_graph_def(graph_def, name="prefix")
  return graph

def main(argv=None):
  cap = cv2.VideoCapture(FLAGS.video_file_name)
  classes = yoloUtils.load_names(FLAGS.class_names)

  # placeholder for detector inputs
  inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

  frozen_graph_file = FLAGS.graph_dir + "/" + FLAGS.graph_name + ".pb"
  graph = load_graph(frozen_graph_file)
  
  x = graph.get_tensor_by_name('prefix/Placeholder:0')
  y = graph.get_tensor_by_name('prefix/detection_boxes:0')

  with tf.Session(graph=graph) as sess:
    startTotal = time.time()
    count = 0
    while(1):
      count += 1
      start = time.time()      
      _, src = cap.read()
      
      src_copy = src
      src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
      src_resized = cv2.resize(src, (FLAGS.size, FLAGS.size))
      detected_boxes = sess.run(y, feed_dict={x: [np.array(src_resized, dtype=np.float32)]})
      filtered_boxes = yoloUtils.non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold, iou_threshold=FLAGS.iou_threshold)
      
      end = time.time()
      ms = (end - start) * 1000
      print "Detection time without drawing: ", ms
      yoloUtils.draw_boxes_cv(filtered_boxes, src_copy, classes, (FLAGS.size, FLAGS.size))

      k = cv2.waitKey(1)
      if k == 27:
        break
  
  endTotal = time.time()
  sTotal = endTotal - startTotal
  sAverage = sTotal / count
  msAverage = sAverage * 1000
  print "Average time per detection: ", msAverage, "ms"
  print "FPS: ", 1 / sAverage, "fps"
  print "Number of operations in the graph: ", len(graph.get_operations())
  # for op in graph.get_operations():
  #   print op.name
  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  tf.app.run()