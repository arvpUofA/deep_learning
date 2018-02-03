class Box():
    def __init__(self,x,y,w,h):
        self.x = x
        self.y = y
        self.w = w
        self.h = h

def overlap(x1, w1, x2, w2):
    left = x1 if x1 > x2 else x2
    r1 = x1 + w1
    r2 = x2 + w2
    right = r1 if r1 < r2 else r2
    return right - left

def box_intersection(box1, box2):
    w = overlap(box1.x, box1.w, box2.x, box2.w)
    h = overlap(box1.y, box1.h, box2.y, box2.h)
    if w < 0 or h < 0:
        return 0
    area = w*h
    return area

def box_union(box1, box2):
    i = box_intersection(box1, box2)
    u = box1.w*box1.h + box2.w*box2.h - i
    return u

def box_iou(box1,box2):
    return box_intersection(box1,box2)/box_union(box1,box2)