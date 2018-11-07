# -*- coding: utf-8 -*-
#!/usr/bin/env python

import numpy as np
import cv2

#cap = cv2.VideoCapture('test.avi')
cap = cv2.VideoCapture(0)#0为设备号，我的笔记本只有一个摄像头

# take first frame of the video
ret,frame = cap.read()

# setup initial location of window
r,h,c,w = 250,90,400,125  # simply hardcoded the values
track_window = (c,r,w,h)

# 设置追踪区域
roi = frame[r:r+h, c:c+w]
# roi区域的hsv图像
hsv_roi =  cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
# 取值hsv值在(0,60,32)到(180,255,255)之间的部分
mask = cv2.inRange(hsv_roi, np.array((0., 60.,32.)), np.array((180.,255.,255.)))
# 计算直方图,参数为 图片(可多)，通道数，蒙板区域，直方图长度，范围
roi_hist = cv2.calcHist([hsv_roi],[0],mask,[180],[0,180])
# 归一化
cv2.normalize(roi_hist,roi_hist,0,255,cv2.NORM_MINMAX)

# 设置终止条件，迭代10次或者至少移动1次
term_crit = ( cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1 )

while(1):
    ret ,frame = cap.read()

    if ret == True:
        # 计算每一帧的hsv图像
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        # 计算反向投影
        dst = cv2.calcBackProject([hsv],[0],roi_hist,[0,180],1)

        # 调用meanShift算法在dst中寻找目标窗口，找到后返回目标窗口
        ret, track_window = cv2.CamShift(dst, track_window, term_crit)

        # Draw it on image
        pts = cv2.boxPoints(ret)
        pts = np.intc(pts)
        img2 = cv2.polylines(frame,[pts],True, 255,2)
        cv2.imshow('img2',img2)

        k = cv2.waitKey(60) & 0xff
        if k == 27:
            break
        else:
            cv2.imwrite(chr(k)+".jpg",img2)


    else:
        break
 
cv2.destroyAllWindows()
cap.release()