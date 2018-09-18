# sadly this does not work as there are unsupported ops for tensorRT
# used to create a tensorRT engine for use in tensorRT 
''' Import TensorRT Modules '''
import tensorrt as trt
import uff
from tensorrt.parsers import uffparser

config = {
  # Where to save models (Tensorflow + TensorRT)
  "frozen_graph_file": "/home/jkc1/deep_learning/tfProjects/yolo_py/graphs/yolov3-tiny/frozen_graph.pb",
  "engine_save_dir": "/home/jkc1/deep_learning/tfProjects/yolo_py/graphs/yolov3-tiny",
  
  # Needed for TensorRT
  "image_dim": 416,  # the image size (square images)
  "inference_batch_size": 1,  # inference batch size
  "input_layer": "prefix/Placeholder",  # name of the input tensor in the TF computational graph
  "out_layer": "detection_boxes",  # name of the output tensorf in the TF conputational graph
  "precision": "fp32",  # desired precision (fp32, fp16)

}

G_LOGGER = trt.infer.ConsoleLogger(trt.infer.LogSeverity.INFO)

def create_and_save_inference_engine():
  # Define network parameters, including inference batch size, name & dimensionality of input/output layers
  INPUT_LAYERS = [config['input_layer']]
  OUTPUT_LAYERS = [config['out_layer']]
  INFERENCE_BATCH_SIZE = config['inference_batch_size']

  INPUT_C = 3
  INPUT_H = config['image_dim']
  INPUT_W = config['image_dim']

  # Load your newly created Tensorflow frozen model and convert it to UFF
  uff_model = uff.from_tensorflow_frozen_model(config['frozen_graph_file'], OUTPUT_LAYERS)

  # Create a UFF parser to parse the UFF file created from your TF Frozen model
  parser = uffparser.create_uff_parser()
  parser.register_input(INPUT_LAYERS[0], (INPUT_C,INPUT_H,INPUT_W),0)
  parser.register_output(OUTPUT_LAYERS[0])

  # Build your TensorRT inference engine
  if(config['precision'] == 'fp32'):
      engine = trt.utils.uff_to_trt_engine(
          G_LOGGER, 
          uff_model, 
          parser, 
          INFERENCE_BATCH_SIZE, 
          1<<20, 
          trt.infer.DataType.FLOAT
      )

  elif(config['precision'] == 'fp16'):
      engine = trt.utils.uff_to_trt_engine(
          G_LOGGER, 
          uff_model, 
          parser, 
          INFERENCE_BATCH_SIZE, 
          1<<20, 
          trt.infer.DataType.HALF
      )
  
  # Serialize TensorRT engine to a file for when you are ready to deploy your model.
  save_path = str(config['engine_save_dir']) + "tf_yolov3tiny_b" \
      + str(INFERENCE_BATCH_SIZE) + "_"+ str(config['precision']) + ".engine"

  trt.utils.write_engine_to_file(save_path, engine.serialize())
  
  print("Saved TRT engine to {}".format(save_path))

create_and_save_inference_engine()