import numpy as np
import cv2
import copy
import tensorflow as tf
import glob
import os
import torch
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import pygame.mixer
from MyVGG import MyVGG
from mouse_event import start_button
from judge import judge
from music import playMusic
from imageCode import *

cap = cv2.VideoCapture(0)

# カスケードファイルのパス
cascade_path = "haarcascade_frontalface_alt.xml"
# カスケード分類器の特徴量取得
cascade = cv2.CascadeClassifier(cascade_path)

param = {"start":True}
cv2.namedWindow('frame')
cv2.setMouseCallback('frame',start_button, param)

titleimg, t_h, t_w, t_x1, t_x2, t_y1, t_y2 = pngRead('image/title.png', 0.65, 60, 250)
Afimg, Af_h, Af_w, Af_x1, Af_x2, Af_y1, Af_y2 = pngRead('image/Arafont.png', 0.65, 280, 10)
startimg, s_h, s_w, s_x1, s_x2, s_y1, s_y2 = pngRead('image/start.png', 0.25, 400, 400)
Arashiimg, A_h, A_w, A_x1, A_x2, A_y1, A_y2 = pngRead('image/AllArashi.png', 0.3, 300, 750)

show_start = 0
while(param["start"]):
    # Capture frame-by-frame
    ret, show = cap.read()
    kernel = np.ones((30,30),np.float32)/900
    show = cv2.filter2D(show,-1,kernel)
    show = cv2.convertScaleAbs(show,alpha = 1.0,beta = 80)
    show = cv2.flip(show, 1)
    show = pngShow(show, titleimg, t_x1, t_x2, t_y1, t_y2)
    show = pngShow(show, Afimg, Af_x1, Af_x2, Af_y1, Af_y2)
    show = pngShow(show, Arashiimg, A_x1, A_x2, A_y1, A_y2)

    if(show_start%10 < 6):
        show = pngShow(show, startimg, s_x1, s_x2, s_y1, s_y2)
    cv2.imshow('frame',show)
    show_start += 1
    cv2.waitKey(10)

s_h, s_w = show.shape[:2]

faceimg, f_h, f_w, _,_,_,_ = pngRead('image/face.png', 0.3, 0, 0)
f_x1, f_x2 = int(s_h/2 - f_h/2)-120, int(s_h/2 + f_h/2)-120, 
f_y1, f_y2 = int(s_w/2 - f_w/2), int(s_w/2 + f_w/2)

kimiha, k_h, k_w, k_x1, k_x2, k_y1, k_y2 = pngRead('image/kimiha.png', 0.25, 540, 50)
da, d_h, d_w, d_x1, d_x2 ,_,_ = pngRead('image/da.png', 0.25, 540, 0)
d_y1, d_y2 = s_w-170-d_w, s_w-170

fon_s = 0.5
Afon, m_h, m_w, m_x1, m_x2, _,_ = pngRead('image/Aiba.png', fon_s, 450, 0)
Nfon,_,_,_,_,_,_ = pngRead('image/Ninomiya.png', fon_s, 0, 0)
Sfon,_,_,_,_,_,_ = pngRead('image/Sakurai.png', fon_s, 0, 0)
Ofon,_,_,_,_,_,_ = pngRead('image/Ohno.png', fon_s, 0, 0)
Mfon,_,_,_,_,_,_ = pngRead('image/Matsumoto.png', fon_s, 0, 0)
m_y1, m_y2 = int(s_w/2 - m_w/2), int(s_w/2 + m_w/2)

face_s = 0.35
Aface, i_h, i_w, i_x1, i_x2, i_y1, i_y2 = pngRead('image/Aface.png', face_s, 30, 920)
Nface,_,_,_,_,_,_ = pngRead('image/Nface.png', face_s, 0, 0)
Sface,_,_,_,_,_,_ = pngRead('image/Sface.png', face_s, 0, 0)
Oface,_,_,_,_,_,_ = pngRead('image/Oface.png', face_s, 0, 0)
Mface,_,_,_,_,_,_ = pngRead('image/Mface.png', face_s, 0, 0)

m_fon = [Afon, Mfon, Nfon, Ofon, Sfon]
m_img = [Aface, Mface, Nface, Oface, Sface]
pre_pred = -1

out_s = 0.2
Aout, o_h, o_w, _,_, o_y1, o_y2 = pngRead('image/Aiba.png', out_s, 0, 30)
Nout,_,_,_,_,_,_ = pngRead('image/Ninomiya.png', out_s, 0, 0)
Sout,_,_,_,_,_,_ = pngRead('image/Sakurai.png', out_s, 0, 0)
Oout,_,_,_,_,_,_ = pngRead('image/Ohno.png', out_s, 0, 0)
Mout,_,_,_,_,_,_ = pngRead('image/Matsumoto.png', out_s, 0, 0)

o_top = 100
o_sp = 60
o0_x1, o0_x2 = o_top, o_top+o_h
o1_x1, o1_x2 = o_top+o_sp, o_top+o_sp+o_h
o2_x1, o2_x2 = o_top+o_sp*2, o_top+o_sp*2+o_h
o3_x1, o3_x2 = o_top+o_sp*3, o_top+o_sp*3+o_h
o4_x1, o4_x2 = o_top+o_sp*4, o_top+o_sp*4+o_h

m_out = [Aout, Mout, Nout, Oout, Sout]

while(True):
    # Capture frame-by-frame
    ret, show = cap.read()
    show = cv2.flip(show, 1)

    # 顔認識
    face_zone = show[f_x1-20:f_x2+20, f_y1-20:f_y2+20]
    faces=cascade.detectMultiScale(face_zone, scaleFactor=1.1, minNeighbors=2, minSize=(40,40))
    show = cv2.convertScaleAbs(show,alpha = 1.0,beta = 80)
    fsave = copy.deepcopy(show)
    
    kernel = np.ones((30,30),np.float32)/900
    show = cv2.filter2D(show,-1,kernel)
    if(len(faces) == 1):
        for x,y,w,h in faces:
            x = f_y1 + x
            y = f_x1 + y
            if( ((s_h/2-120-50) < (y+h/2) < (s_h/2-120+50)) 
            and ((s_w/2 - 50)< (x+w/2) <(s_w/2 + 50)) 
            and (f_w-100) < w < (f_w+100)
            and (f_h-100) < h < (f_h+100)):
                face = fsave[y:y+h, x:x+w]
                rank, opt = judge(face)
            
                result = m_fon[rank[0]]
                fimg = m_img[rank[0]]

                show = SingleColor(show, rank[0])
                show[y:y+h, x:x+w] = face

                show = pngShow(show, result, m_x1, m_x2, m_y1, m_y2)
                show = pngShow(show, kimiha, k_x1, k_x2, k_y1, k_y2)
                show = pngShow(show, da, d_x1, d_x2, d_y1, d_y2)
                show = pngShow(show, fimg, i_x1, i_x2, i_y1, i_y2)

                show = pngShow(show, m_out[rank[0]], o0_x1, o0_x2, o_y1, o_y2)
                show = pngShow(show, m_out[rank[1]], o1_x1, o1_x2, o_y1, o_y2)
                show = pngShow(show, m_out[rank[2]], o2_x1, o2_x2, o_y1, o_y2)
                show = pngShow(show, m_out[rank[3]], o3_x1, o3_x2, o_y1, o_y2)
                show = pngShow(show, m_out[rank[4]], o4_x1, o4_x2, o_y1, o_y2)

                show = showOut(show, opt, o_top+5, o_sp, o_y2+20)

                if( (pre_pred != rank[0]) or (pygame.mixer.music.get_busy() == False) ):
                    playMusic(rank[0])

                if(pre_pred != rank[0]):
                    pre_pred = rank[0]
            else:
                show = pngShow(show, faceimg, f_x1, f_x2, f_y1, f_y2)

    else:
        show = pngShow(show, faceimg, f_x1, f_x2, f_y1, f_y2)
        
    # Display the resulting frame
    cv2.imshow('frame',show)
    if cv2.waitKey(200) & 0xFF == ord('q'):
        break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()