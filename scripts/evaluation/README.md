# Evaluation scripts for object detection

## evaluate_detection.py

Compares yolo formated ground truth annotations with detected boxes.

Usage: `python evaluate_labels.py detected_annotations groundtruth_dir output_dir image_height image_width class_index threshold --write_images` 

detected_annotations should be a txt file and have the following format (with a newline for each detection):
* `image_path item_num class_index x y w h confidence`