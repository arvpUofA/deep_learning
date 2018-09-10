import numpy as np
import tensorflow as tf
import datetime

from utils import yoloUtils

slim = tf.contrib.slim

_BATCH_NORM_DECAY = 0.9
_BATCH_NORM_EPSILON = 1e-05
_LEAKY_RELU = 0.1

_ANCHORS = [(10, 13), (16, 30), (33, 23), (30, 61), (62, 45), (59, 119), (116, 90), (156, 198), (373, 326)]

def darknet53(inputs):
  """
  Builds the Darknet53 model
  """
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 32, 3)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 64, 3, strides=2)
  
  inputs = _darknet53_block(inputs, 32)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 128, 3, strides=2)
  
  for i in range(2):
    inputs = _darknet53_block(inputs, 64)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 256, 3, strides=2)

  for i in range(8):
    inputs = _darknet53_block(inputs, 128)
  route_1 = inputs
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 512, 3, strides=2)

  for i in range(8):
    inputs = _darknet53_block(inputs, 256)
  route_2 = inputs
  inputs = yoloUtils._conv2d_fixed_padding(inputs, 1024, 3, strides=2)

  for i in range(4):
    inputs = _darknet53_block(inputs, 512)
  # no avg pool layer and softmax layer? it is in the paper but what about the config file for yoloV3 (the 'C' implementation)
  return route_1, route_2, inputs

def _darknet53_block(inputs, filters):
  """
  Builds the block with 2 conv layers and a skip-layer connection. Take a look at Darknet-53 blocks from the paper

  Args:
    inputs: 4-D tensor of size [batch_size, ?, ?, ?]
    filters: number of filters to run conv2d on

  Returns:
    A tensor representing the output of the end of the residual block
  """
  shortcut = inputs
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters, 1)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters * 2, 3)

  inputs = inputs + shortcut
  return inputs

def _yolo_block(inputs, filters):
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters, 1)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters * 2, 3)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters, 1)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters * 2, 3)
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters, 1)
  route = inputs
  inputs = yoloUtils._conv2d_fixed_padding(inputs, filters * 2, 3)
  return route, inputs

def yolo_v3(inputs, num_classes, is_training=False, data_format='NCHW', reuse=False):
  """
  Creates YOLO v3 model
  
  Args:
    inputs: 4-D tensor of size [batch_size, hieght, width, channels]
    num_classes: number of classes to predict
    is_training: take a guess.
    data_format: either NCHW or NHWC
    reuse: if the network and variables should be reused

  Returns:

  """
  # needed later on, can this be grabbed elsewhere?
  img_size = inputs.get_shape().as_list()[1:3]

  if data_format == "NCHW":
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  # nomralize to range 0-1
  inputs = inputs * 0.00392156862 # instead of / 255, a tensorRt unsupported realDiv op
  batch_norm_params = {
    'decay': _BATCH_NORM_DECAY,
    'epsilon': _BATCH_NORM_EPSILON,
    'scale': True,
    'is_training': is_training,
    'fused': None, # do we want this?
  }

  with slim.arg_scope([slim.conv2d, slim.batch_norm, yoloUtils._fixed_padding], data_format=data_format, reuse=reuse):
    with slim.arg_scope([slim.conv2d], normalizer_fn=slim.batch_norm, normalizer_params=batch_norm_params,
                        biases_initializer=None, activation_fn=lambda x: tf.nn.leaky_relu(x, alpha=_LEAKY_RELU)):
      with tf.variable_scope('darknet-53'):
        route_1, route_2, inputs = darknet53(inputs)

      with tf.variable_scope("yolo-v3"):
        # scale 1
        route, inputs = _yolo_block(inputs, 512)
        detect_1 = yoloUtils._detection_layer(inputs, num_classes, _ANCHORS[6:9], img_size, data_format)
        detect_1 = tf.identity(detect_1, name='detect_1')
        
        inputs = yoloUtils._conv2d_fixed_padding(route, 256, 1)
        upsample_size = route_2.get_shape().as_list()
        inputs = yoloUtils._upsample(inputs, upsample_size, data_format)
        inputs = tf.concat([inputs, route_2], axis=1 if data_format == 'NCHW' else 3)

        # scale 2
        route, inputs = _yolo_block(inputs, 256)
        detect_2 = yoloUtils._detection_layer(inputs, num_classes, _ANCHORS[3:6], img_size, data_format)
        detect_2 = tf.identity(detect_2, name='detect_2')

        inputs = yoloUtils._conv2d_fixed_padding(route, 128, 1)
        upsample_size = route_1.get_shape().as_list()
        inputs = yoloUtils._upsample(inputs, upsample_size, data_format)
        inputs = tf.concat([inputs, route_1], axis=1 if data_format == 'NCHW' else 3)

        # scale 3
        _, inputs = _yolo_block(inputs, 128)
        detect_3 = yoloUtils._detection_layer(inputs, num_classes, _ANCHORS[0:3], img_size, data_format)
        detect_3 = tf.identity(detect_3, name='detect_3')

        detections = tf.concat([detect_1, detect_2, detect_3], axis=1)
        detections = tf.identity(detections, name='detections')
        return detections
