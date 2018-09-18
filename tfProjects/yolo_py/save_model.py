# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np

from utils import yoloUtils
from models import modelUtils

FLAGS = tf.app.flags.FLAGS

# tf.app.flags.DEFINE_string('class_names', './names/coco.names', 'File with class names')
tf.app.flags.DEFINE_string('class_names', './names/dice.names', 'File with class names')
# tf.app.flags.DEFINE_string('weights_file', './weights/yolov3.weights', 'Binary file with detector weights')
# tf.app.flags.DEFINE_string('weights_file', './weights/yolov3-tiny.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('weights_file', './weights/4classesV3_17000.weights', 'Binary file with detector weights')
tf.app.flags.DEFINE_string('model_name', "yolov3-tiny", "Name of the model to run and save")

tf.app.flags.DEFINE_integer('size', 416, 'Image size')

tf.app.flags.DEFINE_boolean('save', True, 'Save model?')

tf.app.flags.DEFINE_boolean('tensorRT', False, 'TensorRT supported ops only?')

def main(argv=None):

  classes = yoloUtils.load_names(FLAGS.class_names)

  # placeholder for detector inputs
  inputs = tf.placeholder(tf.float32, [None, FLAGS.size, FLAGS.size, 3])

  with tf.variable_scope('detector'):
    # detections = yolo_v3(inputs, len(classes), data_format='NCHW')
    detections = modelUtils.get_model(FLAGS.model_name, inputs, len(classes), data_format='NCHW', tensorRT=FLAGS.tensorRT)
    load_ops = yoloUtils.load_weights(tf.global_variables(scope='detector'), FLAGS.weights_file)

  boxes = yoloUtils.detections_boxes(detections, tensorRT=FLAGS.tensorRT)

  saver = tf.train.Saver()

  with tf.Session() as sess:
    sess.run(load_ops)
    if FLAGS.save:
      print "Saving model"
      save_path = saver.save(sess, "checkpoints/" + FLAGS.model_name + "/" + FLAGS.model_name + ".ckpt")
    else:
      print "BOOLEAN IS FALSE, NOT SAVING"

if __name__ == '__main__':
  tf.app.run()