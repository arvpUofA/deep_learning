"""
evaluates groundtruth with predictions
"""

import os
import argparse
import cv2
from box import *
import glob


if __name__ == '__main__':
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("detected_annotations", help="txt file with detections")
    parser.add_argument("groundtruth_dir", help="folder with ground truth annotations and images")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("image_height", help="resize image height")
    parser.add_argument("image_width", help="resize image width")  
    parser.add_argument("class_index", help="index for class ie, red_buoy=0")
    parser.add_argument("iou_thresh", type=float,help="threshold for correct detection")
    parser.add_argument('--write_image', dest='write_image', action='store_true')    
    
    args = parser.parse_args()

    false_positives = 0
    true_positives = 0
    false_negatives = 0
    image_height = float(args.image_height)
    image_width = float(args.image_width)
    thresh = args.iou_thresh
    write_image = args.write_image

    output_image_path = os.path.join(args.output_dir, "data")
    output_txt_path = os.path.join(args.output_dir, "results.txt")
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    proposals = 0
    total = 0
    correct = 0
    avg_iou = 0

    with open(args.detected_annotations,'r') as anno_file:
        anno_list = anno_file.readlines()
        #annotation dict key=image_path val = list of annocations
        annotations = {}
        for line in anno_list:
            anno = line.split(" ")
            image_path = os.path.join(args.groundtruth_dir, anno[0])
            gt_anno_path = image_path.replace(".jpg",".txt")
            
            conf = float(anno[7])
            if(int(anno[2]) == int(args.class_index) and conf > 0.3):
                x = float(anno[3])
                y = float(anno[4])
                w = float(anno[5])
                h = float(anno[6])
                box_anno = Box(x,y,w,h)
                proposals += 1
                if image_path in annotations:
                    annotations[image_path].append([box_anno,conf])
                else:
                    annotations[image_path] = list()
                    annotations[image_path].append([box_anno,conf])

        print("annotations: ", len(annotations))
        print(proposals)
        
        total_gt = 0
        files = glob.glob(args.groundtruth_dir+'*.txt')
        print(len(files))
        for f in files:
            touch = False
            image_path = f.replace(".txt",".jpg")
            if(write_image):
                img = cv2.imread(image_path)
            gt_file = open(f,'r')
            gt_anno_list = gt_file.readlines()
            total_gt += len(gt_anno_list)
            if image_path in annotations:
                box_list = annotations[image_path]
                for gt_line in gt_anno_list:
                    gt_anno = gt_line.split(" ")
                    #yolo style annotations so need to convert from center point to top left and from ratio
                    gt_x = (float(gt_anno[1]) - float(gt_anno[3])/2) * image_width
                    gt_y = (float(gt_anno[2]) - float(gt_anno[4])/2) * image_height
                    gt_w = float(gt_anno[3]) * image_width
                    gt_h = float(gt_anno[4]) * image_height
                    box_gt = Box(gt_x,gt_y,gt_w,gt_h)
                    best_iou = 0
                    bbox = None
                    for a in box_list:
                        iou = box_iou(a[0],box_gt)
                        if(write_image):
                            img = cv2.rectangle(img,(int(a[0].x),int(a[0].y)),(int(a[0].x+a[0].w),int(a[0].y+a[0].h)),(255,0,0),2)
                        if(iou > best_iou):
                            best_iou = iou
                            bbox = a[0]
                    avg_iou += best_iou
                    total += 1
                    if(best_iou >= thresh):
                        if(write_image):
                            img = cv2.rectangle(img,(int(bbox.x),int(bbox.y)),(int(bbox.x+bbox.w),int(bbox.y+bbox.h)),(0,255,0),2)
                        true_positives += 1
                        false_positives += (len(box_list) - 1)
                    else:
                        false_positives += len(box_list)
            else:
                #print("fn")
                false_negatives += 1
            if(write_image):
                cv2.imwrite(os.path.join(output_image_path,os.path.splitext(os.path.basename(image_path))[0] + '.jpg'),img)
            gt_file.close()
        
        """
        for path, item in annotations.items():
            img = cv2.imread(path)
            #cv2.imshow("")
            gt_file = open(path.replace(".jpg",".txt"),'r')
            gt_anno_list = gt_file.readlines()
            for gt_line in gt_anno_list:
                total += 1
                gt_anno = gt_line.split(" ")
                #yolo style annotations so need to convert from center point to top left and from ratio
                gt_x = (float(gt_anno[1]) - float(gt_anno[3])/2) * image_width
                gt_y = (float(gt_anno[2]) - float(gt_anno[4])/2) * image_height
                gt_w = float(gt_anno[3]) * image_width
                gt_h = float(gt_anno[4]) * image_height
                #img = cv2.rectangle(img,(int(gt_x),int(gt_y)),(int(gt_x+gt_w),int(gt_y+gt_h)),(0,0,255),2)
                box_gt = Box(gt_x,gt_y,gt_w,gt_h)

                best_iou = 0
                bbox = None
                for a in item:
                    iou = box_iou(a[0],box_gt)
                    img = cv2.rectangle(img,(int(a[0].x),int(a[0].y)),(int(a[0].x+a[0].w),int(a[0].y+a[0].h)),(255,0,0),2)
                    if(iou > best_iou):
                        best_iou = iou
                        bbox = a[0]
                avg_iou += best_iou
                if(best_iou > 0.01):
                    correct += 1
                    img = cv2.rectangle(img,(int(bbox.x),int(bbox.y)),(int(bbox.x+bbox.w),int(bbox.y+bbox.h)),(0,255,0),2)
            cv2.imwrite(os.path.join(output_image_path,os.path.splitext(os.path.basename(path))[0] + '.jpg'),img)
            gt_file.close()
        """
        print("Tp:", true_positives)
        print("Fp:", false_positives)
        print("Fn:", false_negatives)
        accuracy = true_positives * 100 / total_gt 
        
        #print(correct)
        #print(total)
        #print(proposals)
        #recall_1 = correct/total * 100
        #prec_1 = correct/proposals * 100
        avg_iou = avg_iou * 100.0 / total
        prec_2 = true_positives/(true_positives+false_positives) * 100
        recall_2 = true_positives/(false_negatives+true_positives) * 100
        
        with open(output_txt_path,'w+') as outfile:
             outfile.write("Recall: {:.2f}, Precision: {:.2f}, Accuracy: {:.2f}, Avg IoU: {:.2f}".format(recall_2,prec_2,accuracy,avg_iou))




        #print("Recall: {}, Precision: {}".format(recall_1,prec_1))
        print("Recall: {}, Precision: {}".format(recall_2,prec_2))
        print("Average IoU: " + str(avg_iou))
        print("Accuracy: ",accuracy)
        print("Miss rate: ",(false_negatives/total_gt) * 100)

        

            
            
           
