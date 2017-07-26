"""
browser yolo dataset
"""

import cv2
import sys
import os
import glob


from pprint import pprint
KEY_RIGHT = 83
KEY_LEFT = 81
KEY_ESC = 27

name_list = ["red_buoy", "green_buoy", "yellow_buoy", "path"]
color_list = [(0,0,255), (255,0,255),(0,255,255), (255,0,0)]

def get_images(path):
    data = None
    images = glob.glob(path+"*.jpg")
    return images

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python browse_dataset.py dataset_root")
        exit(1)
    root_path = sys.argv[1]

    images = get_images(root_path)

    cv2.namedWindow('browse_dataset', cv2.WINDOW_NORMAL)
    cv2.resizeWindow('browse_dataset', 1280, 720)

    index = 0
    print(len(images))
    for image in images:
        print(image)
        anno_path = os.path.join(root_path,image).replace(".jpg",".txt")
        img = cv2.imread(os.path.join(root_path, image))
        print(img.shape)
        image_width = img.shape[1]
        image_height = img.shape[0]
        try:
            with open(anno_path, "r") as anno_file:
                lines = anno_file.readlines()
                for line in lines:
                    
                    line = line.rstrip()
                    bbox = list(map(
                        float,
                        line.split(' ')
                    ))
                    print(line)
                    print(name_list[int(bbox[0])])
                    print(bbox)
                    x = int(bbox[1] * image_width)
                    y = int(bbox[2] * image_height)
                    w = int(bbox[3] * image_width)
                    h = int(bbox[4] * image_height)
                    print(x,y,w,h)
                    print("P1: ", (x - int(w/2), y - int(h/2)))
                    print("P2: ",  (x + int(w/2), y + int(h/2)))
                    
                    cv2.putText(img,name_list[int(bbox[0])], (int(x),int(y - h/2 - 10)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, 255)
                    cv2.rectangle(img, (x - int(w/2), y - int(h/2)), (x + int(w/2), y + int(h/2)), color_list[int(bbox[0])], 1)
            cv2.imshow('browse_dataset', img)
            print("{}/{} - {}".format(index+1, len(images),
                                  image
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
        except FileNotFoundError as e:
            print("File not found")
        

    cv2.destroyAllWindows()