import numpy as np
import cv2
import copy

def pngRead(path, s, x, y):
    img = cv2.imread(path,-1)
    h, w = img .shape[:2]
    img = cv2.resize(img , (int(w*s), int(h*s)))
    h, w = img .shape[:2]

    x1, x2, y1, y2 = x, h+x, y, w+y

    return img,h,w,x1,x2,y1,y2

def pngShow(show, img, x1, x2, y1, y2):
    show[x1:x2, y1:y2] = show[x1:x2, y1:y2] * (1 - img[:, :, 3:] / 255) + \
                                    img[:, :, :3] * (img[:, :, 3:] / 255)
    return show

def SingleColor(show, predicted): 
    h, w = show.shape[:2]
    color = copy.deepcopy(show)

    if(predicted==0): color[:] = (0,123,4)
    elif(predicted==1): color[:] = (167,0,80)
    elif(predicted==2): color[:] = (59,255,241)
    elif(predicted==3): color[:] = (199,0,0)
    else: color[:] = (0,0,255)

    show = cv2.addWeighted(color,0.2,show,0.8,2.2)
    return show

def showOut(show, opt, o_top, o_sp, y):
    o0_x = o_top+o_sp
    o1_x = o_top+o_sp*2
    o2_x = o_top+o_sp*3
    o3_x = o_top+o_sp*4
    o4_x = o_top+o_sp*5
    o_x = [o0_x, o1_x, o2_x, o3_x, o4_x]

    for i in range(5):
        cv2.putText(show,
                text=opt[i]+"%",
                org=(y, o_x[i]),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=2.0,
                color=(0, 0, 0),
                thickness=7,
                lineType=cv2.LINE_4)

    return show