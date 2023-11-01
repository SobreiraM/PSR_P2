#!/usr/bin/env python

########################################################
# PSR - Color Segmenter
#
# Miguel Sobreira, 110045
# Miguel Simões, 118200
# Gonçalo Rodrigues, 98322
########################################################

import cv2
import argparse
import numpy
from functools import partial
import json


def chooseColor(event,x,y,flags,*userdata,img):
    """
    Mouse callback function responsible for the color selection,
    associated to the Color Picker window.
    
    :param img: Image
    :param trackbarInfo: Trackbar information
    """
    B = img[y,x][0]
    G = img[y,x][1]
    R = img[y,x][2]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        print('Selected color (BGR):', int(B), int(G), int(R))
        tresh = 10
        cv2.setTrackbarPos('min B/H', 'window', int(B)-tresh)
        cv2.setTrackbarPos('max B/H', 'window', int(B)+tresh)
        cv2.setTrackbarPos('min G/S', 'window', int(G)-tresh)
        cv2.setTrackbarPos('max G/S', 'window', int(G)+tresh)
        cv2.setTrackbarPos('min R/V', 'window', int(R)-tresh)
        cv2.setTrackbarPos('max R/V', 'window', int(R)+tresh)
        
def onTrackbar(min_B, max_B, min_G, max_G, min_R, max_R, image_hsv):

    min_B = cv2.getTrackbarPos('min B/H', 'window')
    max_B = cv2.getTrackbarPos('max B/H', 'window')
    min_G = cv2.getTrackbarPos('min G/S', 'window')
    max_G = cv2.getTrackbarPos('max G/S', 'window')
    min_R = cv2.getTrackbarPos('min R/V', 'window')
    max_R = cv2.getTrackbarPos('max R/V', 'window')
    thresh = 10
    
    mask = cv2.inRange(image_hsv, (int(min_B-thresh),int(min_G-thresh),int(min_R-thresh)), (int(max_B+thresh),int(max_G+thresh),int(max_R+thresh)))
    cv2.imshow('window', mask)
    
def main():

        # initial setup
    capture = cv2.VideoCapture()
    capture.open('http://192.168.1.23:8000/')
    window_name = 'window'
    cam_name = 'cam'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)
    cv2.namedWindow(cam_name,cv2.WINDOW_AUTOSIZE)

    _, image = capture.read()  # get an image from the camera

    
    cv2.createTrackbar('min B/H', window_name, 0, 255, 
                    lambda x : onTrackbar(x,0,0,0,0,0,image))

    cv2.createTrackbar('max B/H', window_name, 0, 255,
                    lambda x : onTrackbar(0,x,0,0,0,0,image))
    
    cv2.createTrackbar('min G/S', window_name, 0, 255, 
                    lambda x : onTrackbar(0,0,x,0,0,0,image))
    
    cv2.createTrackbar('max G/S', window_name, 0, 255,  
                    lambda x : onTrackbar(0,0,0,x,0,0,image))
    
    cv2.createTrackbar('min R/V', window_name, 0, 255, 
                    lambda x : onTrackbar(0,0,0,0,x,0,image))
    
    cv2.createTrackbar('max R/V', window_name, 0, 255, 
                    lambda x : onTrackbar(0,0,0,0,0,x,image))

    while True:
        
        _, image = capture.read()  # get an image from the camera

        cv2.imshow(cam_name, image)
        cv2.setMouseCallback(cam_name, partial(chooseColor, img=image))
        onTrackbar(0, 0, 0, 0, 0, 0, image)
        
        key = cv2.waitKey(25)
        if key == ord('q'):
            break
        elif key == ord('w'):
            limits = {'min_B': cv2.getTrackbarPos('min B/H', 'window'),
                    'max_B': cv2.getTrackbarPos('max B/H', 'window'),
                    'min_G': cv2.getTrackbarPos('min G/S', 'window'),
                    'max_G': cv2.getTrackbarPos('max G/S', 'window'),
                    'min_R': cv2.getTrackbarPos('min R/V', 'window'),
                    'max_R': cv2.getTrackbarPos('max R/V', 'window')}

            limits_F = open('limits_F.json', 'a')          # Open the file limits_F.json
            limits_F.truncate(0)                          # Clears file from previous runs
            json.dump(limits, limits_F)                   # Saves values from limits
            limits_F.write('\n')
            limits_F.close
            break



if __name__ == "__main__":
    main()
