"""
converts a dataset from JSON format to the kitti format
"""

import os
import argparse
import json
import shutil
import tqdm


def get_json_dataset(json_path):
    """
    create dictonary of json data
    :param json_path: input json file pointing to dataset
    :return: dictionary of json data
    """
    if os.path.isfile(json_path):
        data = None
        with open(json_path) as data_file:
            data = json.load(data_file)
        print("Loaded {} images from {}".format(
            len(data),
            json_path
        ))
        return data
    else:
        print("{} does not exist".format(json_path))
        exit(1)


def create_label_file(labels_path, object_name, item):
    file_name = os.path.join(labels_path,
                             os.path.splitext(os.path.basename(item['image_path']))[0] + '.txt')
    with open(file_name, 'w') as i_file:
        for roi in item['rects']:
            i_file.write("{0} 0.0 0 0.0 {1:.2f} {2:.2f} {3:.2f} {4:.2f} 0.0 0.0 0.0 0.0 0.0 0.0 0.0".format(
                object_name,
                roi['x1'],
                roi['y1'],
                roi['x2'],
                roi['y2']
            ))

if __name__ == '__main__':
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="input json for dataset")
    parser.add_argument("output_folder", help="output folder (will create folder if DNE")
    parser.add_argument("object_name", help="name of object being labeled")
    args = parser.parse_args()

    # get input data
    data = get_json_dataset(args.input_json)

    # get input path
    input_path = os.path.dirname(args.input_json)

    # create output folders
    images_path = os.path.join(args.output_folder, 'images')
    labels_path = os.path.join(args.output_folder, 'labels')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    # create output
    for item in tqdm.tqdm(data):
        # create text file
        create_label_file(labels_path, args.object_name, item)

        # copy images
        shutil.copy2(os.path.join(input_path, item['image_path']), images_path)
