import cv2

def start_button(event,x,y,flags,param):
    if event == cv2.EVENT_LBUTTONDOWN:
        param["start"] = False

