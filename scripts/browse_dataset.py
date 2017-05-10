"""
browser dataset
"""

import cv2
import json
import sys
import os

from pprint import pprint

KEY_RIGHT = 83
KEY_LEFT = 81
KEY_ESC = 27

def get_images(json_file):
    data = None
    with open(json_file) as data_file:
        data = json.load(data_file)
    print("Loaded {} images from {}".format(
        len(data),
        json_file
    ))
    return data

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python browse_dataset.py json_file")
        exit(1)
    json_file = sys.argv[1]

    images = get_images(json_file)

    cv2.namedWindow('browse_dataset', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('browse_dataset', 1280, 720)

    index = 0
    while 1:
        image = cv2.imread(images[index]['image_path'])
        bbox = images[index]['rects'][0]
        cv2.rectangle(image, (bbox['x1'], bbox['y1']), (bbox['x2'], bbox['y2']), (0, 255, 0), 3)

        cv2.imshow('browse_dataset', image)

        print("{}/{} - {}".format(index+1, len(images),
                                  images[index]['image_path'].split('/')[-1]
                                  ))
        c = cv2.waitKey(0)

        if c == KEY_RIGHT:
            index += 1
            if index > len(images) - 1:
                index = len(images) - 1
        elif c == KEY_LEFT:
            index -= 1
            if index < 0:
                index = 0
        elif c == KEY_ESC:
            break

    cv2.destroyAllWindows()