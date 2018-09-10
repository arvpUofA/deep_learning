from yoloV3 import yolo_v3
from yoloV3Tiny import yolo_v3_tiny

def get_model(model_name, inputs, num_classes, is_training=False, data_format='NCHW', reuse=False, tensorRT=False):
  if model_name == "yolov3-tiny":
    return yolo_v3_tiny(inputs, num_classes, is_training, data_format, reuse, tensorRT)
  elif model_name == "yolov3":
    return yolo_v3(inputs, num_classes, is_training, data_format, reuse)
  else:
    print "INCORRECT MODEL NAME HAS BEEN ENTERED"
    exit()