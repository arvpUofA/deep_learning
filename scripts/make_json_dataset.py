"""
make json dataset
Usage: python make_json_dataset.py
"""

import os
import tqdm
import json
import drive_interface
import argparse
import yaml
import random
from PIL import Image


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


def write_folders_to_file(folders, output_file):
    """
    write list of folders to file
    :param folders: list of folders
    :param output_file: output file path
    """
    with open(output_file, 'w+') as o_file:
        for folder in folders:
            o_file.write("{}\n".format(folder['name']))


def read_bbox_file(contents, folder):
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
                'bbox': bbox
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
        os.remove(destination)

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
        print('{}/{} not found.'.format(image_name))
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
        scaling_ratio = scaling_ratios[item['image_name']]
        if scaling_ratio == (None, None):
            print("No scaling ratio for {}".format(item['image_name']))
            exit(1)
        x1 = round(item['bbox'][0] * scaling_ratio[0])
        y1 = round(item['bbox'][1] * scaling_ratio[1])
        w = round(item['bbox'][2] * scaling_ratio[0])
        h = round(item['bbox'][3] * scaling_ratio[1])
        item['bbox'][0] = x1
        item['bbox'][1] = y1
        item['bbox'][2] = w
        item['bbox'][3] = h
        output.append(item)
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


def main():
    # parse cli
    parser = argparse.ArgumentParser()
    parser.add_argument('settings_file', help="settings yaml file")
    parser.add_argument('output_path', help="output path for dataset")
    args = parser.parse_args()

    # parse settings
    settings = read_settings(args.settings_file)

    # initialize drive interface
    drive = drive_interface.DriveInterface('./client_secret.json')
    print("Drive initialized :)")

    # only one object
    if len(settings['objects']) == 1:
        object_name = settings['objects'][0]
        print('Generating data for {}'.format(object_name))

        # get folders
        folders = get_folders(drive, object_name, settings['exclude'])

        # reading bounding-boxes
        print("reading bounding box files")
        for folder in tqdm.tqdm(folders):
            folder['bbox'] = read_bbox_file(
                drive.read_text_file(folder['bbox_file'][0]['id']),
                folder['name']
            )

        # download images
        scaling_ratios = {} # image_name -> scaling ratio map
        data_folder = os.path.join(args.output_path, 'data')
        if not os.path.exists(data_folder):
            os.makedirs(data_folder)
        write_folders_to_file(folders, os.path.join(args.output_path, 'folders.txt'))
        print("Downloading images to {}".format(data_folder))
        for folder in folders:
            print("\nDownloading from {}".format(folder['name']))
            for bbox in tqdm.tqdm(folder['bbox']):
                fx, fy = download_image(
                    drive,
                    bbox['image_name'],
                    folder['id'],
                    os.path.join(data_folder, bbox['image_name']),
                    settings['resolution']
                )
                scaling_ratios[bbox['image_name']] = (fx,fy)

        bboxes = []
        for folder in folders:
            bboxes.extend(folder['bbox'])
        bboxes = scale_bboxes(bboxes, scaling_ratios)
        # split dataset into training and validation
        training_set, validation_set = split_data(bboxes, settings['validation_split'])
        # create to json
        save_to_json(training_set, data_folder,
                     os.path.join(args.output_path, object_name + '_train.json'))
        save_to_json(validation_set, data_folder,
                     os.path.join(args.output_path, object_name + '_val.json'))


if __name__ == '__main__':
    main()
