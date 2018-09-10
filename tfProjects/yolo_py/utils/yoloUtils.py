import numpy as np
import tensorflow as tf
import cv2
import random

slim = tf.contrib.slim

def _get_size(shape, data_format):
  if len(shape) == 4:
    shape = shape[1:]
  return shape[1:3] if data_format == 'NCHW' else shape[0:2]

# well my personal implementation fell apart
def _detection_layer(inputs, num_classes, anchors, img_size, data_format, tensorRT=False):
  num_anchors = len(anchors)
  predictions = slim.conv2d(inputs, num_anchors * (5 + num_classes), 1, stride=1, normalizer_fn=None,
                            activation_fn=None, biases_initializer=tf.zeros_initializer())

  shape = predictions.get_shape().as_list()
  grid_size = _get_size(shape, data_format)
  dim = grid_size[0] * grid_size[1]
  bbox_attrs = 5 + num_classes

  if data_format == 'NCHW':
    predictions = tf.reshape(predictions, [-1, num_anchors * bbox_attrs, dim])
    predictions = tf.transpose(predictions, [0, 2, 1])

  predictions = tf.reshape(predictions, [-1, num_anchors * dim, bbox_attrs])

  stride = (img_size[0] // grid_size[0], img_size[1] // grid_size[1])

  anchors = [(a[0] / stride[0], a[1] / stride[1]) for a in anchors]

  ''' tensorRT does not support split nor does it support slice ....'''
  ''' ##### TensorRT ##### '''
  if tensorRT:
    box_centers = tf.slice(predictions, [0, 0, 0], [-1, -1, 2])
    box_sizes = tf.slice(predictions, [0, 0, 2], [-1, -1, 2])
    confidence = tf.slice(predictions, [0, 0, 4], [-1, -1, 1])
    classes = tf.slice(predictions, [0, 0, 5], [-1, -1, -1])
  else:
    box_centers, box_sizes, confidence, classes = tf.split(predictions, [2, 2, 1, num_classes], axis=-1)
  ''' ##### TensorRT ##### '''


  box_centers = tf.nn.sigmoid(box_centers)
  confidence = tf.nn.sigmoid(confidence)

  ''' ##### TensorRT ##### '''
  if tensorRT:
    ''' create x_offset '''
    # do the first concat as we have to increment grid_line_to_add
    grid_temp = tf.zeros([1, grid_size[0]], dtype=tf.float32)
    grid_line_to_add = tf.ones([1, grid_size[0]], dtype=tf.float32)
    grid_temp = tf.concat([grid_temp, grid_line_to_add], 0)

    # assumption is made that grid_size is symetrical etc [13, 13]
    for i in range(2, grid_size[0]): # start at 2 as we already concatenated once
      # increment by 1
      grid_line_to_add = tf.add(grid_line_to_add, tf.ones([1, grid_size[0]], dtype=tf.float32))
      grid_temp = tf.concat([grid_temp, grid_line_to_add], 0)
    y_offset = tf.reshape(grid_temp, (-1, 1))
    ''' create y_offset end '''
    '''create x_offset'''
    x_offset = tf.constant([i for i in range(grid_size[0])], dtype=tf.float32)
    x_offset_single = x_offset

    for i in range(1, grid_size[0]): # start at 1 as we already have the first x_offset
      x_offset = tf.concat([x_offset, x_offset_single], 0)
    x_offset = tf.reshape(x_offset, (-1, 1))
    ''' create x_offset end '''
  else:
    grid_x = tf.range(grid_size[0], dtype=tf.float32)
    grid_y = tf.range(grid_size[1], dtype=tf.float32)
    a, b = tf.meshgrid(grid_x, grid_y)
    
    x_offset = tf.reshape(a, (-1, 1))
    y_offset = tf.reshape(b, (-1, 1))
  ''' ##### TensorRT ##### '''


  ''' create x_y_offset start '''
  x_y_offset = tf.concat([x_offset, y_offset], axis=-1)
  x_y_offset_single = x_y_offset

  ''' ##### TensorRT ##### '''
  if tensorRT:
    for i in range(num_anchors-1):
      x_y_offset = tf.concat([x_y_offset, x_y_offset_single], 1)
  else:
    x_y_offset = tf.tile(x_y_offset, [1, num_anchors])
  ''' ##### TensorRT ##### '''
  x_y_offset = tf.reshape(x_y_offset, [1, -1, 2])
  ''' create x_y_offset end '''


  box_centers = box_centers + x_y_offset
  box_centers = box_centers * stride

  anchors = tf.convert_to_tensor(anchors, dtype=tf.float32)

  ''' ##### TensorRT ##### '''
  if tensorRT:
    anchors_single = anchors
    for i in range(dim-1):
      anchors = tf.concat([anchors, anchors_single], 0)
  else:
    anchors = tf.tile(anchors, [dim, 1])
  ''' ##### TensorRT ##### '''

  box_sizes = tf.exp(box_sizes) * anchors
  box_sizes = box_sizes * stride
  detections = tf.concat([box_centers, box_sizes, confidence], axis=-1)

  classes = tf.nn.sigmoid(classes)
  predictions = tf.concat([detections, classes], axis=-1)
  return predictions

def _upsample(inputs, out_shape, data_format='NCHW'):
  # tf.image.resize_nearest_neighbor accepts input in format NHWC
  if data_format == 'NCHW':
    inputs = tf.transpose(inputs, [0, 2, 3, 1])

  if data_format == 'NCHW':
    new_height = out_shape[3]
    new_width = out_shape[2]
  else:
    new_height = out_shape[2]
    new_width = out_shape[1]

  inputs = tf.image.resize_nearest_neighbor(inputs, (new_height, new_width))

  # back to NCHW if needed
  if data_format == 'NCHW':
    inputs = tf.transpose(inputs, [0, 3, 1, 2])

  inputs = tf.identity(inputs, name='upsampled')
  return inputs

def _conv2d_fixed_padding(inputs, filters, kernel_size, strides=1):
  if strides > 1:
    inputs = _fixed_padding(inputs, kernel_size)
    padding = 'VALID'
  else:
    padding = 'SAME'
  inputs = slim.conv2d(inputs, filters, kernel_size, stride=strides, padding=padding)
  return inputs

#https://github.com/tensorflow/models/blob/master/official/resnet/resnet_model.py
@tf.contrib.framework.add_arg_scope
def _fixed_padding(inputs, kernel_size, *args, **kwargs):
  """
  Pads the input along the spatial dimensions regardless of input size

  Args:
    inputs: 4-D tensor of size [batch_size, channels, height_in, width_in] for NCHW
                            or [batch_size, height_in, width_in, channels] for NHWC
    kernel_size: Positive integer, size of the kernel to be used in conv2d or max_pool2d.
    data_format: 'NHWC' or 'NCHW'
    mode: The mode for tf.pad (in kwargs)

  Returns:
    A tensor with the same format as the input, with data intact if (kernel_size == 1) else padded    
  """

  mode = kwargs.pop("mode", 'CONSTANT')

  pad_total = kernel_size - 1
  pad_beg = pad_total // 2
  pad_end = pad_total - pad_beg

  if kwargs['data_format'] == 'NCHW':
    padded_inputs = tf.pad(inputs, [[0, 0], [0, 0],
                                    [pad_beg, pad_end], [pad_beg, pad_end]], mode=mode)
  else:
    padded_inputs = tf.pad(inputs, [[0, 0], [pad_beg, pad_end],
                                    [pad_beg, pad_end], [0, 0]], mode=mode)
  return padded_inputs


def load_weights(var_list, weights_file):
  """
  Loads and converts pre-trained weights to something tensorflow can use
  
  Args:
    var_list: list of network variables
    weights_file: name of darknet weight file (example: yolo-arvp.weights)
  
  Returns:
    list of assign ops
  """
  with open(weights_file, "rb") as fp:
    # count 5 if using weights file from pjreddie, 4 if it is from AlexeyAB ?
    _ = np.fromfile(fp, dtype=np.int32, count=4)
    # _ = np.fromfile(fp, dtype=np.int32, count=5)

    weights = np.fromfile(fp, dtype=np.float32)

  ptr = 0
  i = 0
  assign_ops = []
  while i < len(var_list) - 1:
    var1 = var_list[i]
    var2 = var_list[i + 1]
    # do something only if we process conv layer
    if 'Conv' in var1.name.split('/')[-2]:
      # check type of next layer
      if 'BatchNorm' in var2.name.split('/')[-2]:
        # load batch norm params
        gamma, beta, mean, var = var_list[i + 1:i + 5]
        batch_norm_vars = [beta, gamma, mean, var]
        for var in batch_norm_vars:
          shape = var.shape.as_list()
          num_params = np.prod(shape)
          var_weights = weights[ptr:ptr + num_params].reshape(shape)
          ptr += num_params
          assign_ops.append(tf.assign(var, var_weights, validate_shape=True))

        # we move the pointer by 4, because we loaded 4 variables
        i += 4
      elif 'Conv' in var2.name.split('/')[-2]:
        # load biases
        bias = var2
        bias_shape = bias.shape.as_list()
        bias_params = np.prod(bias_shape)
        bias_weights = weights[ptr:ptr + bias_params].reshape(bias_shape)
        ptr += bias_params
        assign_ops.append(tf.assign(bias, bias_weights, validate_shape=True))

        # we loaded 1 variable
        i += 1
      # we can load weights of conv layer
      shape = var1.shape.as_list()
      num_params = np.prod(shape)

      print len(weights) ,ptr, num_params

      var_weights = weights[ptr:ptr + num_params].reshape((shape[3], shape[2], shape[0], shape[1]))
      # remember to transpose to column-major
      var_weights = np.transpose(var_weights, (2, 3, 1, 0))
      ptr += num_params
      assign_ops.append(tf.assign(var1, var_weights, validate_shape=True))
      i += 1

  return assign_ops

def detections_boxes(detections, tensorRT=False):
  """
  Converts center x, center y, width and height values to coordinates of top left and bottom right points.

  Args:
   detections: outputs of YOLO v3 detector of shape (?, 10647, (num_classes + 5))
  
  Returns: 
    converted detections of same shape as input
  """

  ''' tensorRT does not support split nor does it support slice ....'''
  if tensorRT:
    center_x = tf.slice(detections, [0, 0, 0], [-1, -1, 1])
    center_y = tf.slice(detections, [0, 0, 1], [-1, -1, 1])
    width = tf.slice(detections, [0, 0, 2], [-1, -1, 1])
    height = tf.slice(detections, [0, 0, 3], [-1, -1, 1])
    attrs = tf.slice(detections, [0, 0, 4], [-1, -1, -1])

    w2 = width * 0.5 # instead of / 2, a tensorRt unsupported realDiv op
    h2 = height * 0.5 # instead of / 2, a realDiv op
  else:
    center_x, center_y, width, height, attrs = tf.split(detections, [1, 1, 1, 1, -1], axis=-1)
    w2 = width / 2 
    h2 = height /2
    
  x0 = center_x - w2
  y0 = center_y - h2
  x1 = center_x + w2
  y1 = center_y + h2

  boxes = tf.concat([x0, y0, x1, y1], axis=-1)
  detections = tf.concat([boxes, attrs], axis=-1, name="detection_boxes")
  
  return detections

def _iou(box1, box2):
  """
  Computes Intersection over Union value for 2 bounding boxes
  
  Args:
    box1: array of 4 values (top left and bottom right coords): [x0, y0, x1, x2]
    box2: same as box1

  Returns:
   IoU
  """
  b1_x0, b1_y0, b1_x1, b1_y1 = box1
  b2_x0, b2_y0, b2_x1, b2_y1 = box2

  int_x0 = max(b1_x0, b2_x0)
  int_y0 = max(b1_y0, b2_y0)
  int_x1 = min(b1_x1, b2_x1)
  int_y1 = min(b1_y1, b2_y1)

  int_area = (int_x1 - int_x0) * (int_y1 - int_y0)

  b1_area = (b1_x1 - b1_x0) * (b1_y1 - b1_y0)
  b2_area = (b2_x1 - b2_x0) * (b2_y1 - b2_y0)

  # we add small epsilon of 1e-05 to avoid division by 0
  iou = int_area / (b1_area + b2_area - int_area + 1e-05)
  return iou

def non_max_suppression(predictions_with_boxes, confidence_threshold, iou_threshold=0.4):
  """
  Applies Non-max suppression to prediction boxes.

  Args:
    predictions_with_boxes: 3D numpy array, first 4 values in 3rd dimension are bbox attrs, 5th is confidence
    confidence_threshold: the threshold for deciding if prediction is valid
    iou_threshold: the threshold for deciding if two boxes overlap
  
  Returns: 
    dict: class -> [(box, score)]
  """
  conf_mask = np.expand_dims((predictions_with_boxes[:, :, 4] > confidence_threshold), -1)

  predictions = predictions_with_boxes * conf_mask

  result = {}
  image_pred = predictions[0]

  shape = image_pred.shape
  non_zero_idxs = np.nonzero(image_pred) # (2, ?*9) # in this case ? is 4
  image_pred = image_pred[non_zero_idxs] # (?*9, ) flattened values

  image_pred = image_pred.reshape(-1, shape[-1]) # shape[-1] is 9, (?, 9) 
  bbox_attrs = image_pred[:, :5] # (?, 5)
  classes = image_pred[:, 5:] # (?, 4)

  classes = np.argmax(classes, axis=-1) # [1 1 0 0] dice2, dice2, dice1, dice1
  unique_classes = list(set(classes.reshape(-1))) # [0, 1]

  for cls in unique_classes:
    cls_mask = classes == cls # [1 1 0 0] -> [False False True True]
    cls_boxes = bbox_attrs[np.nonzero(cls_mask)] # select id of class that holds true from cls_mask
    #print cls_boxes[:, -1] #is the confidence value [0.7451803 0.334395 ]
    #print cls_boxes[:, -1].argsort()[::-1] # largest to smallest conf, returns id [0, 1] 
    cls_boxes = cls_boxes[cls_boxes[:, -1].argsort()[::-1]] # ^^ 2d array, each with 4 box attrs + conf
    cls_scores = cls_boxes[:, -1] # grab only the confidence values
    cls_boxes = cls_boxes[:, :-1] # grab only the box attrs
    while len(cls_boxes) > 0:
      box = cls_boxes[0] # get first box
      score = cls_scores[0] # get first score
      if not cls in result: # add list for box values and score if it doesnt already exist
        result[cls] = []
      result[cls].append((box, score)) # add the class
      cls_boxes = cls_boxes[1:] # remove the first cls_box (sorted from before)

      # overlap check
      ious = np.array([_iou(box, x) for x in cls_boxes])
      iou_mask = ious < iou_threshold
      cls_boxes = cls_boxes[np.nonzero(iou_mask)]
      cls_scores = cls_scores[np.nonzero(iou_mask)]

  return result

def load_names(file_name):
  names = {}
  with open(file_name) as f:
    for id, name in enumerate(f):
      names[id] = name
  return names

def convert_to_original_size_cv(box, size, original_size):
  ratio = 1.0 * original_size / size # python 2.7 the ratio must be a float type
  box = box.reshape(2, 2) * ratio
  return np.asarray(list(box.reshape(-1)), dtype=np.int32)

def random_color():
    rgbl=[255,0,0]
    random.shuffle(rgbl)
    return tuple(rgbl)

def draw_boxes_cv(boxes, src, cls_names, detection_size):
  for cls, bboxs in boxes.items():
    for box, score in bboxs:
      shape = np.array((src.shape[1], src.shape[0]))
      box = convert_to_original_size_cv(box, np.array(detection_size), shape)
      cv2.rectangle(src, (box[0], box[1]), (box[2], box[3]), random_color(), 2)
      cv2.imshow("detections", src)