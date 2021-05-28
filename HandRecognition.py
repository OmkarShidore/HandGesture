# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 20:31:26 2019

@author: OmkarShidore
git: www.github.com/OmkarShidore
"""
import cv2
import math
import numpy as np
import pandas as pd
video = cv2.VideoCapture(0)
while video.isOpened():
    img1=np.zeros((256,256,3),np.uint8)
    ret, frame = video.read()
    cv2.rectangle(frame, (100, 100), (300, 300), (0, 0, 0), 0)
    crop_image = frame[100:300, 100:300]
    blur = cv2.GaussianBlur(crop_image, (3, 3), 0)
    hsv = cv2.cvtColor(blur, cv2.COLOR_BGR2HSV)
    mask2 = cv2.inRange(hsv, np.array([2, 0, 0]), np.array([20, 255, 255]))
    kernel = np.ones((5, 5))
    dilation = cv2.dilate(mask2, kernel, iterations=1)
    erosion = cv2.erode(dilation, kernel, iterations=1)
    filtered = cv2.GaussianBlur(erosion, (5, 5), 0)
    ret, thresh = cv2.threshold(filtered, 127, 255, 0)
    ret1, thresh2 = cv2.threshold(filtered, 127, 255, cv2.THRESH_BINARY_INV)
    cv2.imshow("Binary", thresh)
    cv2.imshow("Binary-Inverse",thresh2)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    try:
        contour = max(contours, key=lambda x: cv2.contourArea(x))
        x, y, w, h = cv2.boundingRect(contour)
        cv2.rectangle(crop_image, (x, y), (x + w, y + h), (0, 0, 255), 0)
        hull = cv2.convexHull(contour)
        areahull = cv2.contourArea(hull)
        areacnt = cv2.contourArea(contour)
        drawing = np.zeros(crop_image.shape, np.uint8)
        cv2.drawContours(drawing, [contour], -1, (0, 255, 0), 0)
        cv2.drawContours(drawing, [hull], -1, (0, 0, 255), 0)
        hull = cv2.convexHull(contour, returnPoints=False)
        defects = cv2.convexityDefects(contour, hull)
        
        count_defects = 0

        for i in range(defects.shape[0]):
            s, e, f, d = defects[i, 0]
            start = tuple(contour[s][0])
            end = tuple(contour[e][0])
            far = tuple(contour[f][0])
            a = math.sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2)
            b = math.sqrt((far[0] - start[0]) ** 2 + (far[1] - start[1]) ** 2)
            c = math.sqrt((end[0] - far[0]) ** 2 + (end[1] - far[1]) ** 2)
            angle = (math.acos((b ** 2 + c ** 2 - a ** 2) / (2 * b * c)) * 180) / 3.14
            if angle <= 90:
                count_defects += 1
                cv2.circle(crop_image, far, 2, [255,0, 0], -1)
            cv2.line(crop_image, start, end, [0, 255, 0], 2)

        #dected output
        i=90
        j=150
        font_size=4
        font_type=cv2.FONT_HERSHEY_SIMPLEX
        if count_defects == 0:
            cv2.putText(img1, "1", (i, j), font_type, font_size,(255,255,255),2)
        elif count_defects == 1:
            cv2.putText(img1, "2", (i, j), font_type, font_size,(255,255,255), 2)
        elif count_defects == 2:
            cv2.putText(img1, "3", (i, j), font_type, font_size,(255,255,255), 2)
        elif count_defects == 3:
            cv2.putText(img1, "4", (i, j), font_type, font_size,(255,255,255), 2)
        elif count_defects == 4:
            cv2.putText(img1, "5", (i, j), font_type, font_size,(255,255,255), 2)
        else:
            pass
    except:
        pass
    cv2.imshow("Input Frame", frame)
    cv2.imshow("Predicted Output", img1)
    all_image = np.hstack((drawing, crop_image))
    cv2.imshow('Framework', all_image)
    if cv2.waitKey(1) == 27:
        break

video.release()
cv2.destroyAllWindows()
