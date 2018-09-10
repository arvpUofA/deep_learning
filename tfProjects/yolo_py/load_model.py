# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import cv2
import time

from utils import yoloUtils
from models import modelUtils

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('class_names', './names/dice.names', 'File with class names')
tf.app.flags.DEFINE_string('model_name', "yolov3-tiny", "Name of the model to run and save")
tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_float('conf_threshold', 0.25, 'Confidence threshold')
tf.app.flags.DEFINE_float('iou_threshold', 0.4, 'IoU threshold')
tf.app.flags.DEFINE_string('video_file_name', "/home/jkc1/arvp/videos/front-cam-dice-roulette.mp4", "Video to run")

tf.app.flags.DEFINE_boolean('tensorRT', True, 'TensorRT supported ops only?')


def main(argv=None):
  cap = cv2.VideoCapture(FLAGS.video_file_name)
  classes = yoloUtils.load_names(FLAGS.class_names)
  # placeholder for detector inputs
  inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

  with tf.variable_scope('detector'):
    detections = modelUtils.get_model(FLAGS.model_name, inputs, len(classes), data_format='NCHW')

  boxes = yoloUtils.detections_boxes(detections, tensorRT=FLAGS.tensorRT)

  saver = tf.train.Saver()
  with tf.Session() as sess:
    saver.restore(sess, "checkpoints/" + FLAGS.model_name + "/" + FLAGS.model_name + ".ckpt")
    #for op in tf.get_default_graph().get_operations():
    #  print str(op.name) 

    startTotal = time.time()
    count = 0
    while(1):
      count += 1
      start = time.time()      
      _, src = cap.read()
      
      src_copy = src
      src = cv2.cvtColor(src, cv2.COLOR_BGR2RGB)
      src_resized = cv2.resize(src, (FLAGS.size, FLAGS.size))
      detected_boxes = sess.run(boxes, feed_dict={inputs: [np.array(src_resized, dtype=np.float32)]})
      filtered_boxes = yoloUtils.non_max_suppression(detected_boxes, confidence_threshold=FLAGS.conf_threshold, 
      iou_threshold=FLAGS.iou_threshold)
      
      end = time.time()
      ms = (end - start) * 1000
      print "Detection time without drawing: ", ms
      yoloUtils.draw_boxes_cv(filtered_boxes, src_copy, classes, (FLAGS.size, FLAGS.size))

      k = cv2.waitKey(33)
      if k == 27:
        break
  
  endTotal = time.time()
  sTotal = endTotal - startTotal
  sAverage = sTotal / count
  msAverage = sAverage * 1000
  print "Average time per detection: ", msAverage, "ms"
  print "FPS: ", 1/sAverage, "fps"

  cap.release()
  cv2.destroyAllWindows()


if __name__ == '__main__':
  tf.app.run()