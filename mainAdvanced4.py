#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import tkinter as tk
from colorama import Fore, Style
from curses import window
from functools import partial
import json


# chmod +x main.py
# chmod 777 main.py

def arg_function():
    """
    Function to parse arguments from command line.
    """
    parser = argparse.ArgumentParser(description='Definição do modo de teste ' )
    parser.add_argument("-j", "--json", 
                        required=True, 
                        help="Caminho completo para o arquivo JSON")
    parser.add_argument("-pn", "--paint_numbers",
                        required=False, action="store_true",
                        help="Pintura numerada")
    return parser.parse_args()

def cleanHEX(data):
    """
    Remove all non-hexadecimal characters from 'data'.
    
    :param data: String to be cleaned
    """
    validHex = '0123456789ABCDEF'.__contains__
    return ''.join(filter(validHex, data.upper()))

def foreHEX(text, hexcode):
    """
    Print the 'text' in the terminal
    with the color specified by 'hexcode'.
    
    :param text: Text to be printed
    :param hexcode: Hexadecimal color code
    """
    
    hexint = int(cleanHEX(hexcode), 16)
    print("\x1B[38;2;{};{};{}m{}\x1B[0m".format(hexint>>16, hexint>>8&0xFF, hexint&0xFF, text))

def chooseColor(event,x,y,flags,*userdata,img,drawing_data,trackbarInfo):
    """
    Mouse callback function responsible for the color selection,
    associated to the Color Picker window.
    
    :param img: Image
    :param drawing_data: Drawing data
    :param trackbarInfo: Trackbar information
    """
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
    """
    Trackbar callback function responsible for the brightness adjustment,
    associated to the Color Picker window.
    
    :param val: Trackbar value
    :param img: Image
    :param window_name: Window name
    :param drawing_data: Drawing data
    """
    B = drawing_data['selected_color'][0]
    G = drawing_data['selected_color'][1]
    R = drawing_data['selected_color'][2]
    
    drawing_data['color'] = (int(B*val/100),int(G*val/100),int(R*val/100))

def Filter_Desired_Areas(image,limits):
    """
    Function to filter the desired areas of the image.
    
    :param image: Image
    :param limits: Limits
    """
    lower_limits = (limits['min_B'], limits['min_G'], limits['min_R'])
    upper_limits = (limits['max_B'], limits['max_G'], limits['max_R'])
    mask = cv2.inRange(image, lower_limits, upper_limits)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def Find_Largest_Contour(image):
    """
    Function to find the largest contour in the image.
    
    :param image: Image
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Check if countours has any data inside
    if len(contours) == 0:
        return None
    else:
        largest_contour = max(contours, key=cv2.contourArea, default=0)

    return largest_contour

def Draw_Largest_Contour(image, contour):
    """
    Draw the largest contour on the image.
    
    :param image: Image
    :param contour: Contour
    """
    cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)  # Fill the contour

def Process_Frame(frame, limits, last_centroid):
    result_frame_dot, centroid_x, centroid_y = 0,0,0
    
    # Filter desired areas
    filtered_frame = Filter_Desired_Areas(frame,limits)

    # Find the largest contour
    largest_contour = Find_Largest_Contour(filtered_frame)

    # Create a black frame
    result_frame = np.zeros_like(frame)

    # Draw and fill the largest contour on the black frame
    if largest_contour is not None:
        last_centroid = [centroid_x, centroid_y]
        Draw_Largest_Contour(result_frame, largest_contour)

        # Mark the center of largest contour with a circle
        centroid = np.mean(np.argwhere(result_frame),axis=0)
        centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

        result_frame_dot = cv2.circle(frame, (centroid_x,centroid_y), 3, (255,0,0),-1)
    else:
        result_frame_dot = frame
        centroid_x, centroid_y = last_centroid[0], last_centroid[1]

    return result_frame,result_frame_dot, centroid_x, centroid_y
    
def Drawing(drawing_data,blank_image):

    cv2.line(blank_image,(drawing_data['new_x'],drawing_data['new_y']),
              (drawing_data['previous_x'],drawing_data['previous_y']),drawing_data['color'], drawing_data['thickness'])
        
    drawing_data['previous_x'] = drawing_data['new_x']
    drawing_data['previous_y'] = drawing_data['new_y']

def paintNumbers(event,x,y,flag,*userdata,img,drawing_data):
    """
    Mouse callback function responsible for drawing continuous lines,
    associated to the Paint by Numbers window.
    
    :param img: Image
    :param drawing_data: Drawing data
    """
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

def verifyPaintedContours(img):
    """
    Hardcoded function to verify if all numbered contours were filled with the correct color.
    
    :param img: Image
    :return: True if all numbered contours were filled with the correct color, False otherwise
    """
    
    done = True
    
    if (np.array_equal(img[50][50],(180, 180, 180)) and np.array_equal(img[950][50],(180, 180, 180)) and
        np.array_equal(img[950][950],(180, 180, 180)) and np.array_equal(img[50][950],(180, 180, 180))) != True:
        print(Fore.RED + "Wrong color at 4!" + Style.RESET_ALL)
        done = False
        
    if (np.array_equal(img[150][150],(85, 144, 147)) and np.array_equal(img[150][850],(85, 144, 147)) and
        np.array_equal(img[850][150],(85, 144, 147)) and np.array_equal(img[850][850],(85, 144, 147))) != True:
        print(Fore.RED + "Wrong color at 1!" + Style.RESET_ALL)
        done = False
        
    if (np.array_equal(img[650][650],(102, 210, 255)) and np.array_equal(img[300][650],(102, 210, 255))) != True:
        print(Fore.RED + "Wrong color at 2!" + Style.RESET_ALL)
        done = False
        
    if (np.array_equal(img[350][350],(48, 48, 233)) and np.array_equal(img[500][650],(48, 48, 233)) and 
        np.array_equal(img[650][400],(48, 48, 233))) != True:
        print(Fore.RED + "Wrong color at 3!" + Style.RESET_ALL)
        done = False
        
    return done
    
def main():

    args = arg_function()

    # Open the JSON file
    with open(args.json, 'r') as arquivo:
        limits = json.load(arquivo)

    # Print all JSON data in a readable and pleasant way
    print("---------------------------------------------------------------------------------------")
    print("\tmin/Max B:", Fore.BLUE, limits["min_B"], '-',limits["max_B"], Style.RESET_ALL, end=' ')
    print("\tmin/Max G:", Fore.GREEN, limits["min_G"], '-', limits["max_G"], Style.RESET_ALL, end=' ')
    print("\tmin/Max R:", Fore.RED, limits["min_R"], '-', limits["max_R"], Style.RESET_ALL)
    print("---------------------------------------------------------------------------------------")
    # Setup the camera
    capture = cv2.VideoCapture() 
    #capture.open('http://192.168.1.23:8000/')

    # Check if camera opened successfully
    """ if not capture.isOpened():
        print("ERROR.")
        return """

    # Gather the width and height of the camera
    width = int(capture.get(3))     # VAL 3 corresponding to the width of the video
    height = int(capture.get(4))    # VAL 4 corresponding to the height of the video

    # Creating a white image with the same dimension as the camera
    # Create a height by width matrix that is multiplied by 255 to be totally white
    blank_image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8) 

    # Variable Initialization
    root = tk.Tk()
    color_name = 'Color Picker'
    last_centroid = [0,0]
    drawing_data = {'new_x': 0, 'new_y': 0,'previous_x': 0, 'previous_y': 0,'selected_color': (0,0,0), 'color': (0,0,255), 'thickness': 2, 'pencil_down': False}
    # Get the screen width and height for window resizing and positioning
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()
    
    if not args.paint_numbers:
        # Initialization
        rgb_image = cv2.imread('images/RGB_Square.png', cv2.IMREAD_COLOR)
        rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
        
        cv2.namedWindow(color_name, cv2.WINDOW_NORMAL)
        cv2.namedWindow('Processed Frame', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('White Board', cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow('Mask', cv2.WINDOW_KEEPRATIO)
        
        #Resizing and positioning the windows to show the processed frames
        imgSizeW = int(scr_w*0.40)
        imgSizeH = int(scr_h*0.40)
        
        cv2.moveWindow('Processed Frame', int(scr_w/32), int(scr_h/32))
        cv2.resizeWindow('Processed Frame', imgSizeW, imgSizeH)
        
        cv2.moveWindow('White Board', int(scr_w/32 + imgSizeW + (scr_w * 0.05)), int((scr_h/32) + (imgSizeH) + scr_h*0.02))
        cv2.resizeWindow('White Board', imgSizeW, imgSizeH)
        
        cv2.moveWindow('Mask', int(scr_w/32), int((scr_h/32) + (imgSizeH) + 40))
        cv2.resizeWindow('Mask', imgSizeW, imgSizeH)

        # Configure and show the Color Picker window
        rgbSizeW = int(scr_w*0.20)
        rgbSizeH = int(scr_h*0.20)
        cv2.moveWindow(color_name, int(scr_w/32 + (1.25*imgSizeW) + (scr_w * 0.05)), int(scr_h/32))
        cv2.resizeWindow(color_name, rgbSizeW, rgbSizeW)
        cv2.imshow(color_name, rgb_image)
        
        # Add trackbar
        trackbarInfo = {'sliderMax': 100, 'trackbar_name': 'Brightness', 'val': 100, 'window_name': color_name}
        cv2.createTrackbar(trackbarInfo['trackbar_name'], color_name, 0, trackbarInfo['sliderMax'],
                        partial(onColorTrackbar, img = rgb_image, window_name=color_name, drawing_data=drawing_data))
        cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
        onColorTrackbar(trackbarInfo['val'], rgb_image, color_name, drawing_data)

        # Set mouse callback
        cv2.setMouseCallback(color_name, partial(chooseColor, img=rgb_image, drawing_data=drawing_data, trackbarInfo=trackbarInfo))

        while True:
            _, frame = capture.read()

            mask, processed_frame, centroid_x, centroid_y = Process_Frame(frame, limits, last_centroid)
            last_centroid = [centroid_x, centroid_y]
            
            drawing_data['new_x'] = centroid_x
            drawing_data['new_y'] = centroid_y 

            Drawing(drawing_data, blank_image)

            # Display the processed frame
            cv2.imshow('Processed Frame', processed_frame)
            cv2.imshow('White Board', blank_image)
            cv2.imshow('Mask', mask)

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
                blank_image = np.ones((int(height),int(width),3), dtype=np.uint8)*255
            
            elif key == ord('w'):
                print('Saving image as' + Fore.GREEN + ' result.png' + Style.RESET_ALL + ' ...')
                cv2.imwrite('result.png', blank_image)
        
    else:
        
        # Initialization
        paint_image = cv2.imread('images/PaintNumber.png', cv2.IMREAD_COLOR)
        paint_image = cv2.cvtColor(paint_image, cv2.COLOR_BGR2RGB)
        gray = cv2.cvtColor(paint_image, cv2.COLOR_BGR2GRAY)
        
        paint_name = 'Paint by numbers'
        edged_name = 'Contours'
        cv2.namedWindow(paint_name)
        #cv2.namedWindow(edged_name)
        #cv2.namedWindow('Mask')
        
        # Get the height and width of the image, create a blank image with the same dimension
        h,w = gray.shape
        blank_image = np.ones(shape=[h, w, 3], dtype=np.uint8)*255
        
        cv2.setMouseCallback(paint_name, partial(paintNumbers, img=blank_image, drawing_data=drawing_data))

        paint_image_Backup = paint_image.copy()
        
        drawing_data['thickness'] = 20
        
        while True:
            
            # Add the blank canvas and the paint image together
            gray2 = cv2.cvtColor(paint_image, cv2.COLOR_BGR2GRAY)
            thresh2 = cv2.threshold(gray2, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)[1]
            result1 = cv2.bitwise_and(blank_image, blank_image, mask=thresh2)
            result2 = cv2.bitwise_and(paint_image, paint_image, mask=255-thresh2)
            result = cv2.add(result1, result2)
            
            cv2.imshow(paint_name, result) 
            #cv2.imshow(edged_name, edged)
            #cv2.imshow('Mask', returnMask)
            
            key = cv2.waitKey(100)
            
            if key == ord('q'):
                break
            elif key == ord('1'):
                drawing_data['color'] = (85, 144, 147)
            elif key == ord('2'):
                drawing_data['color'] = (102, 210, 255)
            elif key == ord('3'):
                drawing_data['color'] = (48, 48, 233)
            elif key == ord('4'):
                drawing_data['color'] = (180, 180, 180) 
            elif key == ord('c'):
                paint_image = paint_image_Backup.copy()
            elif key == ord('v'):
                print('Verifying if all numbered coutours were filled with the correct color...')
                if verifyPaintedContours(blank_image) == True:
                    print(Fore.GREEN + 'All numbered contours were filled with the correct color!' + Style.RESET_ALL)
            elif key == ord('+'):
                if drawing_data['thickness'] == 60:
                    print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its maximum value.' + Style.RESET_ALL)
                else:
                    drawing_data['thickness'] += 10
                    print('Increasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness'])) 
            elif key == ord('-'):
                if drawing_data['thickness'] == 20:
                    print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its minimum value.' + Style.RESET_ALL)
                else:
                    drawing_data['thickness'] -= 10
                    print('Decreasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness']))                            

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()