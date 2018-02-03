"""
converts a dataset from JSON format to the yolo format
"""

import os
import argparse
import json
import shutil
import tqdm
import cv2


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


def create_label_file(file_name, item, index, image_height, image_width, ry, rx):
    dw = 1.0/image_width
    dh = 1.0/image_height

    # see https://github.com/AlexeyAB/darknet#how-to-train-to-detect-your-custom-objects
    with open(file_name, 'w') as i_file:
        for roi in item['rects']:
            x = ((roi['x1'] + roi['x2'])/2.0 - 1) * rx * dw
            y = ((roi['y1'] + roi['y2'])/2.0 - 1) * ry * dh
            w = ((roi['x2'] - roi['x1'])) * rx * dw
            h = ((roi['y2'] - roi['y1'])) * ry * dh
            
            i_file.write("{0} {1:.8f} {2:.8f} {3:.8f} {4:.8f}".format(
                index,
                x,
                y,
                w,
                h
            ))

if __name__ == '__main__':
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("input_json", help="input json for dataset")
    parser.add_argument("output_folder", help="output folder (will create folder if DNE")
    parser.add_argument("object_name", help="name of object being labeled")
    parser.add_argument("image_height", help="resize image height")
    parser.add_argument("image_width", help="resize image width")  
    parser.add_argument("yolo_names", help="yolo names file path")   
    
    args = parser.parse_args()

    index = -1
    with open(args.yolo_names,'r') as name_file:
        lines = name_file.readlines()
        for line in lines:
            newline = line.rstrip()
            lines[lines.index(line)] = newline
        try:
            index = lines.index(args.object_name)
        except ValueError:
            print("ValueError: Object name does not match any names in yolo names file")
            quit()
            

    # get input data
    data = get_json_dataset(args.input_json)

    # get input path
    input_path = os.path.dirname(args.input_json)

    # create output folders
    images_path = os.path.join(args.output_folder, 'data')
    labels_path = os.path.join(args.output_folder, 'data')
    if not os.path.exists(images_path):
        os.makedirs(images_path)
    if not os.path.exists(labels_path):
        os.makedirs(labels_path)

    # create imdb file
    output_txt = os.path.split(args.input_json)[1].replace(".json",".txt")
    output_txt = os.path.join(args.output_folder,output_txt)
    imdb = open(output_txt,'w+')

    # create output
    for item in tqdm.tqdm(data):
        # update imdb
        file_name = os.path.join(labels_path,
                             os.path.splitext(os.path.basename(item['image_path']))[0] + '.txt')
        image_file_name = os.path.join(labels_path,
                             os.path.splitext(os.path.basename(item['image_path']))[0] + '.jpg')
        imdb.write(image_file_name + "\n")


        img = cv2.imread(os.path.join(input_path, item['image_path']))
        rx = float(args.image_width) / img.shape[1]
        ry = float(args.image_height) / img.shape[0]
        img = cv2.resize(img,(int(args.image_width),int(args.image_height)))
        cv2.imwrite(image_file_name,img)
        # create text file
        create_label_file(file_name, item, index, float(args.image_height),float(args.image_width), ry,rx)

        
        # copy images
        #shutil.copy2(os.path.join(input_path, item['image_path']), images_path)
    imdb.close()
