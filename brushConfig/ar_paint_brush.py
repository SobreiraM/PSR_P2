#!/usr/bin/env python3
# chmod +x main.py
# chmod 777 main.py

import argparse
import cv2
import numpy as np
import tkinter as tk
from colorama import Fore, Style
from curses import window
from functools import partial

def cleanHEX(data):
    """
    Remove all non-hexadecimal characters from 'data'
    
    :param data: String to be cleaned
    """
    validHex = '0123456789ABCDEF'.__contains__
    return ''.join(filter(validHex, data.upper()))

def foreHEX(text, hexcode):
    """
    Print the 'text' in the terminal
    with the color specified by 'hexcode'
    
    :param text: Text to be printed
    :param hexcode: Hexadecimal color code
    """
    
    hexint = int(cleanHEX(hexcode), 16)
    print("\x1B[38;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text))

def mouseCallback(event,x,y,flag,*userdata,img,drawing_data):
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_data['pencil_down'] = True
        # print(Fore.BLUE + 'pencil_down set to True' + Style.RESET_ALL)
        
    elif event == cv2.EVENT_LBUTTONUP: 
        drawing_data['pencil_down'] = False
        # print(Fore.RED + 'pencil_down released' + Style.RESET_ALL)

    if drawing_data['pencil_down'] == True:
        cv2.line(img, (drawing_data['previous_x'], drawing_data['previous_y']), (x,y), drawing_data['color'], drawing_data['thickness']) 

    drawing_data['previous_x'] = x
    drawing_data['previous_y'] = y

def chooseColor(event,x,y,flags,*userdata,img,drawing_data,trackbarInfo):
    B = img[y,x][0]
    G = img[y,x][1]
    R = img[y,x][2]
    
    if event == cv2.EVENT_LBUTTONDOWN:
        drawing_data['selected_color'] = (int(B),int(G),int(R))
        drawing_data['color'] = (int(B),int(G),int(R))
        print('Setting pencil to the selected color (BGR):', end = ' ')
        foreHEX(drawing_data['color'], '#%02x%02x%02x' % (R,G,B))
        cv2.setTrackbarPos(trackbarInfo['trackbar_name'], trackbarInfo['window_name'], 100)

def onColorTrackbar(val, img, window_name, drawing_data):
    B = drawing_data['selected_color'][0]
    G = drawing_data['selected_color'][1]
    R = drawing_data['selected_color'][2]
    
    drawing_data['color'] = (int(B*val/100),int(G*val/100),int(R*val/100))

def main():
    # Variable Initialization
    root = tk.Tk()
    window_name = 'Webcam'
    color_name = 'Color Picker'
    ip_address = 'http://192.168.1.23:8000/'
    drawing_data = {'pencil_down': False, 'previous_x': 0, 'previous_y': 0,'selected_color': (0,0,0), 'color': (0,0,255), 'thickness': 2}
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()
    
    # Initialization
    capture = cv2.VideoCapture()
    rgb_image = cv2.imread('images/RGB_Square.png', cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(color_name, cv2.WINDOW_NORMAL)

    # Open the camera and get the image size, with it make an empty canvas
    capture.open(ip_address)
    _, _ = capture.read()
    w = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
    h = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
    img = np.ones((int(h),int(w),3), dtype=np.uint8)*255
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Configure and show the Color Picker window
    cv2.imshow(color_name, rgb_image)
    rgbSizeW = int(scr_w*0.20)
    rgbSizeH = int(scr_w*0.20)
    cv2.moveWindow(color_name, int(scr_w - 1.5 * rgbSizeW), int(scr_h - 2.5 * rgbSizeH))
    cv2.resizeWindow(color_name, rgbSizeW, rgbSizeH)

    # Add trackbar
    trackbarInfo = {'sliderMax': 100, 'trackbar_name': 'Brightness', 'val': 100, 'window_name': color_name}
    cv2.createTrackbar(trackbarInfo['trackbar_name'], color_name, 0, trackbarInfo['sliderMax'],
                       partial(onColorTrackbar, img = rgb_image, window_name=color_name, drawing_data=drawing_data))
    cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
    onColorTrackbar(trackbarInfo['val'], rgb_image, color_name, drawing_data)

    # Set all mouse callbacks
    cv2.setMouseCallback(window_name, partial(mouseCallback, img=img, drawing_data=drawing_data))
    cv2.setMouseCallback(color_name, partial(chooseColor, img=rgb_image, drawing_data=drawing_data, trackbarInfo=trackbarInfo))
    
    # Visualization
    while True:
        cv2.imshow(window_name, img)
        key = cv2.waitKey(50)
        
        if key == ord('q'):
            break
        
        elif key == ord('r'):
            print('Setting pencil to' + Fore.RED + ' red color' + Style.RESET_ALL)
            drawing_data['color'] = (0, 0, 255)
            drawing_data['selected_color'] = (0, 0, 255)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
       
        elif key == ord('g'):
            print('Setting pencil to' + Fore.GREEN + ' green color' + Style.RESET_ALL)
            drawing_data['color'] = (0, 255, 0)
            drawing_data['selected_color'] = (255, 0, 0)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
        
        elif key == ord('b'):
            print('Setting pencil to' + Fore.BLUE + ' blue color' + Style.RESET_ALL)
            drawing_data['color'] = (255, 0, 0)
            drawing_data['selected_color'] = (255, 0, 0)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
        
        elif key == ord('+'):
            if drawing_data['thickness'] == 20:
                print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its maximum value.' + Style.RESET_ALL)
            else:
                drawing_data['thickness'] += 1
                print('Increasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness']))
                
        elif key == ord('-'):
            if drawing_data['thickness'] == 1:
                print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its minimum value.' + Style.RESET_ALL)
            else:
                drawing_data['thickness'] -= 1
                print('Decreasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness']))    
        
        elif key == ord('c'):
            print('Clearing image')
            img = np.ones((int(h),int(w),3), dtype=np.uint8)*255
            cv2.setMouseCallback(window_name, partial(mouseCallback, img=img, drawing_data=drawing_data))
        
        elif key == ord('w'):
            print('Saving image as' + Fore.GREEN + ' result.png' + Style.RESET_ALL + ' ...')
            cv2.imwrite('result.png', img)

    # Termination
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()