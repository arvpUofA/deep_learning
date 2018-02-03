"""
make yolo dataset
Usage: python make_yolo_dataset.py /path/to/datasetX.yaml /path/to/output/datasetXXX /path/to/labels

- Examples yaml files can be are 
"""

import os
import tqdm
import json
import argparse
import yaml
import random
from PIL import Image
import glob
import shutil

duplicate_start = 2
num_duplicates = 1

all_objects = ["red_buoy", "green_buoy", "yellow_buoy", "path", "inverted_gate","first_gate"]

def read_settings(settings_file):
    """
    read yaml settings file
    FORMAT
    ------
    objects:
    - red_buoy
    merge_all: false
    merge_name: buoy
    resolution: 1280x720
    validation_split:0.2
    exclude:
    - 2015-06-21_UOFAWEST_GOPRO_7
    --------
    :param settings_file: path to settings file
    :return: {'objects':[], resolution:(w,h), merge_all:false, merge_name:'buoy', validation_split:0.0, exclude:[]}
    """
    with open(settings_file, 'r') as stream:
        try:
            settings = yaml.load(stream)
            settings['resolution'] = tuple(int(x) for x in settings['resolution'].strip().split('x'))
            return settings
        except yaml.YAMLError as e:
            print(e)
            exit(1)

def write_folders_to_file(folders, output_file):
    """
    write list of folders to file
    :param folders: list of folders
    :param output_file: output file path
    """
    with open(output_file, 'w+') as o_file:
        for folder in folders:
            o_file.write("{}\n".format(folder))

def split_data(dataset, validation_ratio):
    """
    split dataset into training and validation set 
    :param dataset: dataset list
    :param validation_ratio: ratio of images set aside for validation
    :return: training,validation
    """
    # shuffle and split data
    random.shuffle(dataset)
    print("{} bounding boxes read and shuffled.".format(len(dataset)))
    print("{} training-validation split".format(validation_ratio))
    validation_images = int(validation_ratio * len(dataset))
    training_set = dataset[validation_images:]
    print("{} images for training".format(len(training_set)))
    validation_set = dataset[:validation_images]
    print("{} images for validation".format((len(validation_set))))
    return training_set, validation_set


def scale_bboxes(bboxes, scaling_ratios):
    """
    scales bounding boxes using scaling ratios dictionary
    :param bboxes: list of bounding boxes
    :param scaling_ratios: dictionary with image_names -> scaling ratios
    :return: scaled bounding boxes
    """
    output = []
    for item in bboxes:
        if(item['image_name'] in scaling_ratios):        
            scaling_ratio = scaling_ratios[item['image_name']]
            if scaling_ratio == (None, None):
                print("No scaling ratio for {}".format(item['image_name']))
                exit(1)
            #print("@scale_bboxes")
            #print(scaling_ratio)
            #print(item['bbox'])
            x1 = round(item['bbox'][0] * scaling_ratio[0])
            y1 = round(item['bbox'][1] * scaling_ratio[1])
            w = round(item['bbox'][2] * scaling_ratio[0])
            h = round(item['bbox'][3] * scaling_ratio[1])
            item['bbox'][0] = x1
            item['bbox'][1] = y1
            item['bbox'][2] = w
            item['bbox'][3] = h
            output.append(item)
        else:
            bboxes.remove(item)
    return output

def save_to_yolo(yolo_dict,data_folder,image_files, tag_file, resolution, class_id):
    """
    save dataset to json file
    :param dataset: dataset
    :param data_folder: folder containing images
    :param json_file: path to json file
    """
    output = []
    #yolo_dict = {}
    count = 0
    print(tag_file)
    with open(tag_file,'r') as f:
        #image_path = os.path.join('data', item['image_name'])
        anno_list = list(f)
        #print(len(anno_list))
        #print(len(image_files))
        for i, line in enumerate(anno_list):
            bbox = list(map(
                int,
                line.split(' ')
            ))

            if len(bbox) == 4 and sum(bbox) > 0:
                image_path = ""
                try:
                    image_path = image_files[i]
                except IndexError as e:
                    print(e)
                    return count, yolo_dict
                count += 1

                img = Image.open(image_path)
                width, height = img.size
                scale = [0,0]
                scale[0] = float(resolution[0])/float(width) 
                scale[1] = float(resolution[1])/float(height)

                x1 = max(bbox[0], 0) * scale[0]  # x
                x2 = (bbox[0] + bbox[2]) * scale[0]  # x + w
                y1 = max(bbox[1], 0) * scale[1] # y
                y2 = (bbox[1] + bbox[3]) * scale[1]  # y + h
                #print("P1: ",x1,y1)
                #print("P2: ",x2,y2)
                #print("Res: ",resolution)
                yolo_x = (float(x1 + x2)/2.0 - 1.0) / float(resolution[0])
                yolo_y = (float(y1 + y2)/2.0 - 1.0) / float(resolution[1])
                yolo_w = float(x2 - x1) / float(resolution[0])
                yolo_h = float(y2 - y1) / float(resolution[1])
                #print("yolo: ",yolo_x, yolo_y)
                #print("w,h: ",yolo_w,yolo_h)

                anno_string = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(class_id,yolo_x,yolo_y,yolo_w,yolo_h)
                if image_path in yolo_dict:
                    yolo_dict[image_path].append(anno_string)
                else:
                    yolo_dict[image_path] = list()
                    yolo_dict[image_path].append(anno_string)
    #print(len(yolo_dict))
    return count,yolo_dict
        
        

def main():
    # parse cli
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help="settings yaml file")
    parser.add_argument('output_path', help="output path for dataset")
    parser.add_argument('tagged_images', help="root directory")
    args = parser.parse_args()

    # parse settings
    settings = read_settings(args.settings_file)
    directories = glob.glob(args.tagged_images + "/*/")

    print(len(directories))
    dataset_name = os.path.basename(os.path.normpath(args.output_path))
    exclude_list = settings['exclude']
    object_names = settings['objects']
    class_count = {name:0 for name in object_names}
    resolution = settings['resolution']
    yolo_file = os.path.join(args.output_path, dataset_name + '_train.txt')

    data_folder = os.path.join(args.output_path, 'data')
    if not os.path.exists(data_folder):
        os.makedirs(data_folder)
    else:
        answer = input("Are you sure you want to overwrite this dataset: Yes or No")
        print("you entered", answer)
        if(answer == "Yes"):
            print("Creating new dataset...")
            shutil.rmtree(args.output_path)
            os.makedirs(data_folder)
        else:
            print("Cancelling dataset creation...")
            return

    scaling_ratios = {}
    write_folders_to_file(directories, os.path.join(args.output_path, 'folders.txt'))
    

    exclude_string = ""

    for folder in directories:
        if(os.path.basename(os.path.normpath(folder)) in exclude_list):
            exclude_string += '\nExcluded: {}'.format(folder)
            continue
        save_images = False
        yolo_dict = {}
        
        print("\nChecking for tags in {}".format(folder))
        tag_files = glob.glob(folder + "/*.txt")
        image_files = sorted(glob.glob(folder + "/*.jpg"))
        for tag_file in tag_files:
            name = os.path.basename(tag_file)
            for o_name in object_names: 
                if(o_name in all_objects):
                    if(o_name in name):
                        class_id = all_objects.index(o_name)
                        save_images = True
                        num_labels,yolo_dict = save_to_yolo(yolo_dict,data_folder,image_files,tag_file, resolution, class_id)
                        class_count[o_name] += num_labels
                else:
                    print("{} is not in all_objects list".format(o_name))
        if len(yolo_dict) > 0:
            for key, value in yolo_dict.items():
                out_txt = os.path.join(data_folder,os.path.basename(key)).replace(".jpg",".txt")
                with open(out_txt, 'w+') as o_file:
                    for anno in value:
                        o_file.write(anno)
            print("Saved {} annotations to {}".format(len(yolo_dict), data_folder))
        
        with open(yolo_file, "a+") as out_file:
            for line in tqdm.tqdm(list(yolo_dict.keys())):
                img_name = os.path.basename(line)
                path = os.path.join("data", img_name)
                out_file.write(path + "\n")
                img = Image.open(line)
                o_width, o_height = img.size
                if (o_width, o_height) != resolution:
                    # scale image
                    img = img.resize(resolution)
                    img.save(os.path.join(data_folder,img_name))
                else:
                    shutil.copy2(line, data_folder)
            print("Saved image list to {}".format(yolo_file))
    
    print(exclude_string)
    print(class_count)
  
if __name__ == '__main__':
    main()
