"""
make json dataset
Usage: python make_json_dataset.py
"""

import sys
import os
import tqdm
import json
import drive_interface


def main():
    # read parameters
    object_name = input('Object_name: ')    # e.g. red_buoy
    output_folder = input('Output folder: ')
    data_folder = os.path.join(output_folder, 'data')
    print("Saving data to {}".format(data_folder))
    json_file = os.path.join(output_folder, object_name + '.json')
    print("Saving json to {}".format(json_file))
    data_list = []

    # initialize drive interface
    drive = drive_interface.DriveInterface('./client_secret.json')

    # make output folder
    if not os.path.exists(output_folder):
        os.makedirs(data_folder)    # automatically makes output_folder + data_folder

    # get all video folders inside tagged_videos folder
    video_folders = drive.get_files(
        mime_type='application/vnd.google-apps.folder',
        parents=drive_interface.FOLDER_IDS['tagged_videos']
    )

    # process each folder
    no_of_folders = len(video_folders)
    folder_index = 1
    for folder in video_folders:
        print('Processing {} of {} folders'.format(folder_index, no_of_folders))
        # get bbox file
        file = drive.get_files(
            file_name=object_name+'.txt',
            mime_type='text',
            parents=folder['id']
        )
        if file:
            # bbox file for object found
            print('Found {} in {}. Adding to dataset'.format(object_name, folder['name']))
            # read text file
            file_id = 1
            bboxes_found = 0
            for line in tqdm.tqdm(drive.read_text_file(file[0]['id']).strip().split('\n')):
                bbox = list(map(int, line.split(' ')))
                if len(bbox) == 4 and sum(bbox) > 0: # bbox present
                    bboxes_found += 1
                    # download image
                    image_file_name = '{0}-{1:05d}.jpg'.format(folder['name'], file_id)
                    image_path = os.path.join(data_folder, image_file_name)
                    if not os.path.isfile(image_path): # image does not exist locally
                        image_remote = drive.get_files(
                            file_name=image_file_name,
                            mime_type='image',
                            parents=folder['id']
                        )
                        if len(image_remote) == 1:  # image found remotely. Downloading now
                            drive.download_file(image_remote[0]['id'], image_path, progress=False)
                        else:
                            print('{}/{} not found.'.format(folder['name'], image_file_name))
                            exit(1)

                    # add to json list
                    data_list.append({
                        'image_path': image_path,
                        'rects': [
                            {
                                'x1': max(bbox[0], 0),      # x
                                'x2': bbox[0] + bbox[2],    # x + w
                                'y1': max(bbox[1], 0),      # y
                                'y2': bbox[1] + bbox[3]     # y + h
                            }
                        ]
                    })
                file_id += 1
            print("Downloaded {} images".format(bboxes_found))
        folder_index += 1

    # output data_list to json
    if data_list:
        with open(json_file, 'w+') as o_file:
            o_file.write(json.dumps(data_list))
        print("Data saved to json")

if __name__ == '__main__':
    main()
