"""
evaluates groundtruth with predictions
"""

import os
import argparse
import cv2
from util.box import *
import glob

def get_spaced_colors(n):
    max_value = 16581375 #255**3
    interval = int(max_value / n)
    colors = [hex(I)[2:].zfill(6) for I in range(0, max_value, interval)]
    
    return [(int(i[:2], 16), int(i[2:4], 16), int(i[4:], 16)) for i in colors]


if __name__ == '__main__':
    # parse command line parameters
    parser = argparse.ArgumentParser()
    parser.add_argument("detected_annotations", help="txt file with detections")
    parser.add_argument("groundtruth_dir", help="folder with ground truth annotations and images")
    parser.add_argument("output_dir", help="output directory")
    parser.add_argument("image_height", help="resize image height")
    parser.add_argument("image_width", help="resize image width")  
    parser.add_argument("class_names", help="file with class names")
    parser.add_argument("iou_thresh", type=float,help="threshold for correct detection")
    parser.add_argument('--write_image', dest='write_image', action='store_true')    
    
    args = parser.parse_args()


    image_height = float(args.image_height)
    image_width = float(args.image_width)
    thresh = args.iou_thresh
    write_image = args.write_image
    

    classes = []
    with open(args.class_names,"r") as class_file:
        for i in class_file.readlines():
            classes.append(i.rstrip())
    class_anno_list = [ {} for i in classes]
    class_eval = [ { "false_positives": 0, "false_negatives": 0, "true_positives": 0,"avg_iou":0,"total": 0,"class_total":0} for i in classes]
    draw_boxes = {}
    colors = get_spaced_colors(len(classes))

    output_image_path = os.path.join(args.output_dir, "data" + str(int(thresh*100)))
    output_txt_path = os.path.join(args.output_dir, "results" + str(int(thresh*100)) + ".txt")
    if not os.path.exists(output_image_path):
        os.makedirs(output_image_path)

    proposals = 0
    total = 0
    correct = 0
    avg_iou = 0

    with open(args.detected_annotations,'r') as anno_file:
        anno_list = anno_file.readlines()

        #darknet produces duplicates when doing multiclass detection so only keep unique
        #TODO: debug why this is happening 
        anno_list = list(set(anno_list))
        #annotation dict key=image_path val = list of annocations
        
        for line in anno_list:
            anno = line.split(" ")
            image_path = os.path.join(args.groundtruth_dir, anno[0])
            gt_anno_path = image_path.replace(".jpg",".txt")
            
            conf = float(anno[7])
            if(conf > 0.3):
                annotations = class_anno_list[int(anno[2])]
                x = float(anno[3])
                y = float(anno[4])
                w = float(anno[5])
                h = float(anno[6])
                box_anno = Box(x,y,w,h)
                proposals += 1
                if image_path not in annotations:
                    class_anno_list[int(anno[2])][image_path] = list()
                        
                class_anno_list[int(anno[2])][image_path].append([box_anno,conf])

                #print("annotations class: " + str(int(anno[2])) + " " + classeslen(annotations))
        print(proposals)

        for i, d in enumerate(class_anno_list):
            print("{}: {}".format(classes[i],len(d)))
            
        
        
    
        files = glob.glob(os.path.join(args.groundtruth_dir,'*.txt'))
        print(args.groundtruth_dir)
        print(len(files))
        
        for i,class_name in enumerate(classes):
            false_positives = 0
            true_positives = 0
            false_negatives = 0
            total_gt = 0
            avg_iou = 0
            total = 0
            annotations = class_anno_list[i]
            class_eval[i]["total"] = len(files)
            for f in files:
                touch = False
                image_path = f.replace(".txt",".jpg")
                #if(write_image):
                #    img = cv2.imread(image_path)
                gt_file = open(f,'r')
                gt_anno_list = gt_file.readlines()
                #total_gt += len(gt_anno_list)
                for gt_line in gt_anno_list:
                    gt_anno = list(map(
                        float,
                        gt_line.split(' ')
                    ))
                    if(gt_anno[0] == i):
                        total += 1
                        if image_path in class_anno_list[i]:
                            box_list = class_anno_list[i][image_path]
                            #yolo style annotations so need to convert from center point to top left and from ratio
                            gt_x = ((gt_anno[1]) - (gt_anno[3])/2) * image_width
                            gt_y = ((gt_anno[2]) - (gt_anno[4])/2) * image_height
                            gt_w = (gt_anno[3]) * image_width
                            gt_h = (gt_anno[4]) * image_height
                            box_gt = Box(gt_x,gt_y,gt_w,gt_h)
                            best_iou = 0
                            bbox = None
                            for a in box_list:
                                iou = box_iou(a[0],box_gt)
                                #if(write_image):
                                    #img = cv2.rectangle(img,(int(a[0].x),int(a[0].y)),(int(a[0].x+a[0].w),int(a[0].y+a[0].h)),(255,0,0),2)
                                if(iou > best_iou):
                                    best_iou = iou
                                    bbox = a[0]
                                if(image_path not in draw_boxes):
                                    draw_boxes[image_path] = list()

                                draw_boxes[image_path].append([(int(a[0].x),int(a[0].y)),(int(a[0].x+a[0].w),int(a[0].y+a[0].h)),i])

                            avg_iou += best_iou
                            #total += 1
                            if(best_iou >= thresh):
                                #if(write_image):
                                #img = cv2.rectangle(img,(int(bbox.x),int(bbox.y)),(int(bbox.x+bbox.w),int(bbox.y+bbox.h)),(0,255,0),2)
                                #if(image_path not in draw_boxes):
                                #    draw_boxes[image_path] = list()

                                #draw_boxes[image_path].append([(int(bbox.x),int(bbox.y)),(int(bbox.x+bbox.w),int(bbox.y+bbox.h)),i])

                                true_positives += 1.0
                                false_positives += (len(box_list) - 1)
                            else:
                                false_positives += len(box_list)
                        
                        else:
                            false_negatives += 1.0
                #if(write_image):
                    #cv2.imwrite(os.path.join(output_image_path,os.path.splitext(os.path.basename(image_path))[0] + '.jpg'),img)
                gt_file.close()

            class_eval[i]["false_positives"] = false_positives
            class_eval[i]["false_negatives"] = false_negatives
            class_eval[i]["true_positives"] = true_positives
            class_eval[i]["avg_iou"] = avg_iou * 100 / total
            class_eval[i]["class_total"] = total
            
        if(write_image):
            print("draw boxes: " + str(len(draw_boxes)))

            for f in files:
                image_path = f.replace(".txt",".jpg")
                img = cv2.imread(image_path)
                
                if(image_path in draw_boxes):
                    value = draw_boxes[image_path]
                    for v in value:
                        c_id = v[2]
                        img = cv2.rectangle(img,v[0],v[1],colors[c_id],2)
                        corner = v[0]
                        cv2.putText(img,classes[c_id], (corner[0],corner[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colors[c_id])
                cv2.imwrite(os.path.join(output_image_path,os.path.splitext(os.path.basename(image_path))[0] + '.jpg'),img)
        
        with open(output_txt_path,'w+') as outfile:
            outfile.write("IoU threshold: {}\n".format(thresh))
            print("\n##################################")
            print("Evaluation")
            print("##################################")
            print("\nIoU threshold: {}".format(thresh))
            mAP = 0
            for i, results in enumerate(class_eval):
                false_positives = class_eval[i]["false_positives"] 
                false_negatives = class_eval[i]["false_negatives"]
                true_positives = class_eval[i]["true_positives"]
                accuracy = true_positives * 100.0 / class_eval[i]["class_total"] 
                avg_iou = class_eval[i]["avg_iou"] 

                prec = true_positives/float(true_positives+false_positives) * 100.0
                recall = true_positives/(false_negatives+true_positives) * 100.0
                outfile.write("{}: Recall: {:.2f}, Precision: {:.2f}, Accuracy: {:.2f}, Avg IoU: {:.2f}\n".format(classes[i],recall,prec,accuracy,avg_iou))
                
                mAP += prec
                print("\n#########################")
                print(classes[i] + ":")
                print("false_pos", false_positives)
                print("false_negs", false_negatives)
                print("true_pos", true_positives)
                print("class_total", class_eval[i]["class_total"] )
                print("Recall: {}, Precision: {}".format(recall,prec))
                print("Average IoU: " + str(avg_iou))
                print("Accuracy: ",accuracy)
                print("Miss rate: ",(false_negatives/class_eval[i]["class_total"]) * 100)
                print("#########################")

            mAP = mAP / len(classes)
            outfile.write("mAP: {}".format(mAP))
            print("mAP: {}\n".format(mAP))


        #print("Recall: {}, Precision: {}".format(recall_1,prec_1))


        

            
            
           
