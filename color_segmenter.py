import cv2
import argparse
import numpy
from functools import partial
import json
import pprint



def onTrackbar(min_B, max_B, min_G, max_G, min_R, max_R, image_hsv):

    min_B = cv2.getTrackbarPos('min B/H', 'window')
    max_B = cv2.getTrackbarPos('max B/H', 'window')
    min_G = cv2.getTrackbarPos('min G/S', 'window')
    max_G = cv2.getTrackbarPos('max G/S', 'window')
    min_R = cv2.getTrackbarPos('min R/V', 'window')
    max_R = cv2.getTrackbarPos('max R/V', 'window')


    mask = cv2.inRange(image_hsv, (min_B,min_G,min_R), (max_B,max_G,max_R))
    cv2.imshow('window', mask)



"""
    print(str(min_B), end=' ')
    print(str(max_B), end=' ')
    print(str(min_G), end=' ')
    print(str(max_G), end=' ')
    print(str(min_R), end=' ')
    print(str(max_R))
"""

def main():

        # initial setup
    capture = cv2.VideoCapture(0)
    window_name = 'window'
    cv2.namedWindow(window_name,cv2.WINDOW_AUTOSIZE)

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

        onTrackbar(0, 0, 0, 0, 0, 0, image)
        
        key = cv2.waitKey(25)
        if key == ord(' '):
            break
        elif key == ord('w'):
            limits = {'min_B': cv2.getTrackbarPos('min B/H', 'window'),
                    'max_B': cv2.getTrackbarPos('max B/H', 'window'),
                    'min_G': cv2.getTrackbarPos('min G/S', 'window'),
                    'max_G': cv2.getTrackbarPos('max G/S', 'window'),
                    'min_R': cv2.getTrackbarPos('min R/V', 'window'),
                    'max_R': cv2.getTrackbarPos('max R/V', 'window')}
            
            limits_F = open('limits_F.txt', 'a')          # Open the file limits_F.txt 
            limits_F.truncate(0)                          # Clears file from previous runs
            json.dump(limits, limits_F)                   # Saves values from limits
            limits_F.write('\n')
            limits_F.close
            break



if __name__ == "__main__":
    main()