"""
make json dataset
Usage: python make_json_dataset.py
"""

import os
import tqdm
import json
from util.drive_interface import *
import argparse
import yaml
import random
from PIL import Image
import cv2


duplicate_start = 2
num_duplicates = 1

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


def get_folders(drive, object_name, exclude):
    """
    gets folder containing object's ROI
    :param drive: drive interface
    :param object_name: name of object
    :param exlude list
    :return: list of folders dictionary with key(bbox_file) for the bbox file
    """
    print("Retrieving folder information")
    # get all video folders inside tagged_videos folder
    video_folders = drive.get_files(
        mime_type='application/vnd.google-apps.folder',
        parents=drive_interface.FOLDER_IDS['tagged_videos']
    )
    object_folders = []
    exclude_string = ''
    # find folders containing ROI files for specified object
    for folder in tqdm.tqdm(video_folders):
        if folder['name'] in exclude:
            exclude_string += '\nexcluding: {}'.format(folder['name'])
            continue
        # get bbox file
        file = drive.get_files(
            file_name=object_name + '.txt',
            mime_type='text',
            parents=folder['id']
        )
        if file:
            # bbox file for object found
            folder['bbox_file'] = file
            object_folders.append(folder)
    if exclude_string != '':
        print(exclude_string)
    print("{} folders found containing {} bbox information".format(len(object_folders), object_name))
    return object_folders

def get_folders2(drive, object_names, exclude):
    """
    gets folder containing object's ROI
    :param drive: drive interface
    :param object_name: name of object
    :param exlude list
    :return: list of folders dictionary with key(bbox_file) for the bbox file
    """
    print("Retrieving folder information")
    # get all video folders inside tagged_videos folder
    video_folders = drive.get_files(
        mime_type='application/vnd.google-apps.folder',
        parents=drive_interface.FOLDER_IDS['tagged_videos']
    )
    object_folders = []
    exclude_string = ''
    add_string = ''
    no_data_string = ''
    # find folders containing ROI files for specified object
    
    for folder in tqdm.tqdm(video_folders):
        add_folder = False
        if folder['name'] in exclude:
            exclude_string += '\nexcluding: {}'.format(folder['name'])
            continue
        for name in object_names:
            # get bbox file
            file = drive.get_files(
                file_name=name + '.txt',
                mime_type='text',
                parents=folder['id']
            )
            if file:
                add_folder = True
                print('Generating data for {}'.format(name))
                # bbox file for object found
                
                folder[name + '_bbox_file'] = file
                folder[name + '_bbox'] = read_bbox_file(
                    drive.read_text_file(folder[name + '_bbox_file'][0]['id']),
                    folder['name'],
                    name
                )
                #object_folders.append(folder)
            
            # Now check if there are any duplicate objects in folder eg. path2.txt
            if(name == "path"):
                for i in range(duplicate_start, duplicate_start + num_duplicates):
                    file = drive.get_files(
                        file_name=name + str(i) + '.txt',
                        mime_type='text',
                        parents=folder['id']
                    )
                    if file:
                        add_folder = True
                        print('Generating data for {}'.format(name + str(i)))
                        # bbox file for object found
                        folder[name + str(i) + '_bbox_file'] = file
                        folder[name + str(i) + '_bbox'] = read_bbox_file(
                            drive.read_text_file(folder[name + str(i) + '_bbox_file'][0]['id']),
                            folder['name'],
                            name
                        )
                        #object_folders.append(folder)
        if add_folder:
            object_folders.append(folder)
            add_string += '\nincluding: {}'.format(folder['name'])
        else:
            no_data_string += '\nno labels for: {}'.format(folder['name'])
        #break
    if exclude_string != '':
        print(exclude_string)
    print(add_string)
    if no_data_string != '':
        print(no_data_string)
    #print("{} folders found containing {} bbox information".format(len(object_folders), object_name))
    print(len(object_folders))
    return object_folders


def write_folders_to_file(folders, output_file):
    """
    write list of folders to file
    :param folders: list of folders
    :param output_file: output file path
    """
    with open(output_file, 'w+') as o_file:
        for folder in folders:
            o_file.write("{}\n".format(folder['name']))


def read_bbox_file(contents, folder, class_name):
    """
    read bbox file and returns list of bboxes as dictionary with keys image_name, bbox
    :param contents: 
    :param folder: 
    :return: 
    """
    lines = contents.strip().split('\n')
    bboxes = []
    for i, line in enumerate(lines):
        bbox = list(map(
            int,
            line.split(' ')
        ))
        if len(bbox) == 4 and sum(bbox) > 0:  # bbox present
            file_id = i+1
            image_name = '{0}-{1:05d}.jpg'.format(folder, file_id)
            bboxes.append({
                'image_name': image_name,
                'bbox': bbox,
                'class_name': class_name
            })
    return bboxes


def download_image(drive, image_name, folder_id, destination, resolution):
    """
    download image to local disk
    :param drive: drive_interface
    :param image_name: image name
    :param folder_id: folder id of parent folder
    :param destination: path to destination on disk
    :param resolution: tuple with resolution for new image (width,height)
    :return fx, fy ratio of new resolution to original resolution (or None)
    """
    if os.path.isfile(destination):
        #os.remove(destination)
        #print(destination + " is already downloaded")
        # image will already get resized
        #img = cv2.imread(destination)
        
        #img = Image.open(destination)
        
        #o_width, o_height = img.size
        #if (o_width, o_height) != resolution:
            # scale image
        #    img = img.resize(resolution)
        #    img.save(destination)
        #    return float(resolution[0])/float(o_width), float(resolution[1])/float(o_height)
        #else:
            #cv2.write(destination,img)
        return -1, -1 # fx, fy
    #else:
    image_remote = drive.get_files(
        file_name=image_name,
        mime_type='image',
        parents=folder_id
    )
    if len(image_remote) == 1:  # image found remotely. Downloading now
        # drive.download_file(image_remote[0]['id'], destination, progress=False)
        image = drive.read_image_file(image_remote[0]['id'])
        o_width, o_height = image.size
        if (o_width, o_height) != resolution:
            # scale image
            image = image.resize(resolution)
            image.save(destination)
            return float(resolution[0])/float(o_width), float(resolution[1])/float(o_height)
        else:
            image.save(destination)
            return 1, 1 # fx, fy

    else:
        print('{} not found.'.format(image_name))
        return None, None


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


def save_to_json(dataset, data_folder, json_file):
    """
    save dataset to json file
    :param dataset: dataset
    :param data_folder: folder containing images
    :param json_file: path to json file
    """
    output = []
    for item in dataset:
        output.append({
            'image_path': os.path.join('data', item['image_name']),
            'rects': [
                {
                    'x1': max(item['bbox'][0], 0),  # x
                    'x2': item['bbox'][0] + item['bbox'][2],  # x + w
                    'y1': max(item['bbox'][1], 0),  # y
                    'y2': item['bbox'][1] + item['bbox'][3]  # y + h
                }
            ]
        })
    if output:
        json_file = os.path.join(json_file)
        with open(json_file, 'w+') as o_file:
            o_file.write(json.dumps(output))
        print("Saved {} images to {}".format(len(output), json_file))

def save_to_yolo(dataset, data_folder, txt_file, resolution, object_names):
    """
    save dataset to json file
    :param dataset: dataset
    :param data_folder: folder containing images
    :param json_file: path to json file
    """
    output = []

    yolo_dict = {}

    for item in dataset:
        image_path = os.path.join('data', item['image_name'])

        #x1 = max(item['bbox'][0], 0)
        #y1 = max(item['bbox'][1], 0)
        #w = item['bbox'][2]
        #h = item['bbox'][3]
        #print("@save_to_yolo")
        #print(item['bbox'])
        x1 = max(item['bbox'][0], 0)  # x
        x2 = item['bbox'][0] + item['bbox'][2]  # x + w
        y1 = max(item['bbox'][1], 0)  # y
        y2 = item['bbox'][1] + item['bbox'][3]  # y + h
        #print("P1: ",x1,y1)
        #print("P2: ",x2,y2)
        #print("Res: ",resolution)
        yolo_x = (float(x1 + x2)/2.0 - 1.0) / float(resolution[0])
        yolo_y = (float(y1 + y2)/2.0 - 1.0) / float(resolution[1])
        yolo_w = float(x2 - x1) / float(resolution[0])
        yolo_h = float(y2 - y1) / float(resolution[1])
        #print("yolo: ",yolo_x, yolo_y)
        #print("w,h: ",yolo_w,yolo_h)
        class_id = object_names.index(item['class_name'])

        anno_string = "{} {:.4f} {:.4f} {:.4f} {:.4f}\n".format(class_id,yolo_x,yolo_y,yolo_w,yolo_h)
    
        if image_path in yolo_dict:
            yolo_dict[image_path].append(anno_string)
        else:
            yolo_dict[image_path] = list()
            yolo_dict[image_path].append(anno_string)
    
    
    if len(yolo_dict) > 0:
        for key, value in yolo_dict.items():
            out_txt = os.path.join(data_folder,os.path.basename(key)).replace(".jpg",".txt")
            with open(out_txt, 'w') as o_file:
                for anno in value:
                    o_file.write(anno)
        print("Saved {} images to {}".format(len(yolo_dict), data_folder))
    
    with open(txt_file, "a+") as out_file:
        for line in list(yolo_dict.keys()):
            out_file.write(line + "\n")
        print("Saved image list to {}".format(txt_file))

def main():
    # parse cli
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help="settings yaml file")
    parser.add_argument('output_path', help="output path for dataset")
    parser.add_argument('format', help="yolo or json")
    args = parser.parse_args()

    # parse settings
    settings = read_settings(args.settings_file)

    # initialize drive interface
    drive = drive_interface.DriveInterface('./client_secret.json')
    
    print("Drive initialized :)")
    class_count = {}
    dataset_name = os.path.basename(os.path.normpath(args.output_path))
    # only one object
    if len(settings['objects']) > 0:
        object_names = settings['objects']

        # get folders
        folders = get_folders2(drive, object_names, settings['exclude'])

        # reading bounding-boxes
        #print("reading bounding box files")
        #for folder in tqdm.tqdm(folders):
        #    folder['bbox'] = read_bbox_file(
        #        drive.read_text_file(folder['bbox_file'][0]['id']),
        #        folder['name']
        #)

        
        # download images
        scaling_ratios = {} # image_name -> scaling ratio map
        data_folder = os.path.join(args.output_path, 'data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        write_folders_to_file(folders, os.path.join(args.output_path, 'folders.txt'))
        print("Downloading images to {}".format(data_folder))
        for folder in folders:
            print("\nDownloading from {}".format(folder['name']))
            for name in object_names:
                if (name+'_bbox') in folder:    
                    for bbox in tqdm.tqdm(folder[name + '_bbox']):
                        fx, fy = download_image(
                            drive,
                            bbox['image_name'],
                            folder['id'],
                            os.path.join(data_folder, bbox['image_name']),
                            settings['resolution']
                        )

                        if(fx == -1 and fy == -1 ):
                            folder[name + '_bbox'].remove(bbox)
                        else:
                            scaling_ratios[bbox['image_name']] = (fx,fy)    
                if(name == "path"):
                    for i in range(duplicate_start, duplicate_start + num_duplicates):
                        if (name+str(i)+'_bbox') in folder:
                            for bbox in tqdm.tqdm(folder[name + str(i) + '_bbox']):
                                fx, fy = download_image(
                                    drive,
                                    bbox['image_name'],
                                    folder['id'],
                                    os.path.join(data_folder, bbox['image_name']),
                                    settings['resolution']
                                )
                                if(fx == -1 and fy == -1 ):
                                    folder[name + str(i) + '_bbox'].remove(bbox)
                                else:
                                    scaling_ratios[bbox['image_name']] = (fx,fy)   
                                
            bboxes = []
            
            for folder in folders:
                for name in object_names:
                    if (name+'_bbox') in folder:
                        if(name in class_count):
                            class_count[name] += len(folder[name + '_bbox'])
                        else:
                            class_count[name] = len(folder[name + '_bbox'])
                        bboxes.extend(folder[name + '_bbox'])
                    if(name == "path"):
                        for i in range(duplicate_start, duplicate_start + num_duplicates):
                            if (name+str(i)+'_bbox') in folder:
                                if(name in class_count):
                                    class_count[name] += len(folder[name + str(i) + '_bbox'])
                                else:
                                    class_count[name] = len(folder[name + str(i) + '_bbox'])
                                bboxes.extend(folder[name + str(i) + '_bbox'])
            bboxes = scale_bboxes(bboxes, scaling_ratios)
            print(len(bboxes))
            save_to_yolo(bboxes, data_folder,
                     os.path.join(args.output_path, dataset_name + '_train.txt'), settings['resolution'],
                     object_names)
        print(class_count)
                
       
        print(len(bboxes))
        # split dataset into training and validation
        training_set, validation_set = split_data(bboxes, settings['validation_split'])
        # create to json

        

        """if(args.format == "json"):
            save_to_json(training_set, data_folder,
                     os.path.join(args.output_path, dataset_name + '_train.json'))
            save_to_json(validation_set, data_folder,
                     os.path.join(args.output_path, dataset_name + '_val.json'))
        elif(args.format == "yolo"):
            save_to_yolo(training_set, data_folder,
                     os.path.join(args.output_path, dataset_name + '_train.txt'), settings['resolution'],
                     object_names)
            save_to_yolo(validation_set, data_folder,
                     os.path.join(args.output_path, dataset_name + '_val.txt'), settings['resolution'],
                     object_names)"""


if __name__ == '__main__':
    main()
