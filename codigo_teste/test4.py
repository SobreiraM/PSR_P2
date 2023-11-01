#!/usr/bin/env python

import argparse
import cv2
import numpy as np
import tkinter as tk
from colorama import Fore, Style
from curses import window
from functools import partial
import json
import math



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
    parser.add_argument('-usp',
                        '--use_shake_prevention',
                        action='store_true',
                        help='Add a functionality named ' + Fore.YELLOW + 'shake prevention' + Style.RESET_ALL)
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

def chooseColor(event,x,y,flags,*userdata,img,drawing_data,trackbarInfo, shapes_data):
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
        drawing_data['color'] = shapes_data['color'] = (int(B),int(G),int(R))
        print('Setting pencil to the selected color (BGR):', end = ' ')
        foreHEX(drawing_data['color'], '#%02x%02x%02x' % (R,G,B))
        cv2.setTrackbarPos(trackbarInfo['trackbar_name'], trackbarInfo['window_name'], 100)

def onColorTrackbar(val, img, window_name, drawing_data, shapes_data):
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
    
    shapes_data['color'] = drawing_data['color'] = (int(B*val/100),int(G*val/100),int(R*val/100))
    

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
    
def Drawing(drawing_data,blank_image,var_m):

    if var_m != 1:

        cv2.line(blank_image,(drawing_data['new_x'],drawing_data['new_y']),
                (drawing_data['previous_x'],drawing_data['previous_y']),drawing_data['color'], drawing_data['thickness'])
            
        drawing_data['previous_x'] = drawing_data['new_x']
        drawing_data['previous_y'] = drawing_data['new_y']



def Draw_On_Camera(frame,drawing):
    gray = cv2.cvtColor(drawing, cv2.COLOR_BGR2GRAY)
    _, mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)

    mask = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
    inverted_mask = cv2.bitwise_not(mask)
    non_white_part = cv2.bitwise_and(frame, mask)
    rest_of_image2 = cv2.bitwise_and(drawing, inverted_mask)
    result = cv2.add(non_white_part, rest_of_image2)
    return result

   
def Circles(shapes_data):
    radius = int(np.sqrt((shapes_data['start_x'] - shapes_data['current_x'])**2 +
                          (shapes_data['start_y'] - shapes_data['current_y'])**2))
    
    key = cv2.waitKey(1)
    if key == ord('a'):
        shapes_data['circle'] = False
        cv2.circle(shapes_data['image'], (shapes_data['start_x'], shapes_data['start_y']), radius,
                    shapes_data['color'], 3)
    elif key == 27:                         # Press 'Esc' to cancel
        shapes_data['rectangle'] = False

def Rectangles(shapes_data):
    key = cv2.waitKey(1)
    if key == ord('a'):
        shapes_data['rectangle'] = False
        cv2.rectangle(shapes_data['image'], (shapes_data['start_x'], shapes_data['start_y']),
                   (shapes_data['current_x'],shapes_data['current_y']), shapes_data['color'], 3)
    elif key == 27:                         # Press 'Esc' to cancel
        shapes_data['rectangle'] = False

def sp(limit, drawing_data, blank_image,var_m):

        # Calculate the distance between the centroid detected and the previous centroid
        distance = math.sqrt(((drawing_data['new_x']  - drawing_data['previous_x']) ** 2) + ((drawing_data['new_y'] - drawing_data['previous_y']) ** 2))

        # If the distance is above a certain limit
        if distance > limit:
            
            drawing_data['previous_x'] = drawing_data['new_x'] 
            drawing_data['previous_y'] = drawing_data['new_y'] 

        else:
            Drawing(drawing_data, blank_image,var_m) 


def click_events(event, x, y, flags, params):
    args = arg_function()
    if event == cv2.EVENT_LBUTTONDOWN:
        # Obtenha a imagem e as coordenadas anteriores dos parâmetros
        blank_image, prev_coords, drawing_data,limit = params

        if prev_coords[0] is not None:
            # Calcule a distância entre as coordenadas anteriores e atuais
            distance = math.sqrt((x - prev_coords[0][0])**2 + (y - prev_coords[0][1])**2)
            distance_int = int(distance)

            if args.use_shake_prevention:

                if distance < limit:  # Limite de 50 pixels
                    # Desenhe uma linha da coordenada anterior para a coordenada atual
                    cv2.line(blank_image, prev_coords[0], (x, y), (0, 0, 255), 8)
                    drawing_data['new_x'] = x
                    drawing_data['new_y'] = y
                    cv2.imshow('White Board', blank_image)
                else: 
                    drawing_data['previous_x'] = prev_coords[0][0]
                    drawing_data['previous_y'] = prev_coords[0][1] 
                    print(f'Distance {Fore.RED}greater {Style.RESET_ALL}than the predefined usp limit -> {Fore.RED}{distance_int} > {limit}{Style.RESET_ALL}') 


            else:
                
                cv2.line(blank_image, prev_coords[0], (x, y), (0, 0, 255), 8)
                drawing_data['new_x'] = x
                drawing_data['new_y'] = y
                cv2.imshow('White Board', blank_image)
                

        prev_coords[0] = (x, y)
        drawing_data['previous_x'] = prev_coords[0][0]
        drawing_data['previous_y'] = prev_coords[0][1]                     


    


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
    capture = cv2.VideoCapture(0) 
    #capture.open('http://192.168.1.23:8000/')

    # Check if camera opened successfully
    if not capture.isOpened():
        print("ERROR.")
        return

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
    drawing_data = {'new_x': 0, 'new_y': 0,'previous_x': 0, 'previous_y': 0,'selected_color': (0,0,0),
                    'color': (0,0,255), 'thickness': 2}
    
    shapes_data = {'start_x': 0, 'start_y': 0, 'current_x': 0, 'current_y': 0, 'rectangle': False, 'circle': False,
                'image': blank_image, 'color': (0,0,0), 'thickness' : 2}

    # Get the screen width and height for window resizing and positioning
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()

    #SP and verifications parameters
    prev_coords = [None]    
    var_m = 0
    limit =70
    
    # Initialization
    rgb_image = cv2.imread('images/RGB_Square.png', cv2.IMREAD_COLOR)
    #rgb_image = cv2.imread('RGB_Square.png', cv2.IMREAD_COLOR)
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
        partial(onColorTrackbar, img = rgb_image, window_name=color_name, drawing_data=drawing_data, shapes_data=shapes_data))
    cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
    onColorTrackbar(trackbarInfo['val'], rgb_image, color_name, drawing_data, shapes_data)

    # Set mouse callback
    cv2.setMouseCallback(color_name, partial(chooseColor, img=rgb_image, drawing_data=drawing_data, trackbarInfo=trackbarInfo,
                                             shapes_data=shapes_data))
    
    
    
    while True:

        _, frame = capture.read()

        mask, processed_frame, centroid_x, centroid_y = Process_Frame(frame, limits, last_centroid)
        last_centroid = [centroid_x, centroid_y]
        
        drawing_data['new_x'] = centroid_x
        drawing_data['new_y'] = centroid_y 


     
        if shapes_data['circle'] == True:               # Condition to keep the shape being drawn if
            shapes_data['current_x'] = centroid_x       # the user has pressed 'o' for circles or 
            shapes_data['current_y'] = centroid_y       # 's' for rectangles and if this process hasn't
            Circles(shapes_data)                        # ended yet

        elif shapes_data['rectangle'] == True:
            shapes_data['current_x'] = centroid_x
            shapes_data['current_y'] = centroid_y
            Rectangles(shapes_data)

        elif args.use_shake_prevention:
        
            sp(limit, drawing_data, blank_image,var_m)
            
            shapes_data['start_x'] = centroid_x         # This will refresh the coordinates of where to start drawing the shapes
            shapes_data['start_y'] = centroid_y         # if the user enters the drawing shapes mode
    
            
        else:                                           # If none of the previous conditions are true,
            Drawing(drawing_data, blank_image,var_m)          # the program shoud follow its natural procedure to draw lines
            
            shapes_data['start_x'] = centroid_x         # This will refresh the coordinates of where to start drawing the shapes
            shapes_data['start_y'] = centroid_y         # if the user enters the drawing shapes mode

  

        blended_frame = Draw_On_Camera(frame, blank_image)
           
    
        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)
        cv2.imshow('Mask', mask)
        cv2.imshow('Blended Frame', blended_frame)
        cv2.imshow('White Board', blank_image)  
        
        
        

        key = cv2.waitKey(1)

        if key == ord('q'):
            break
        
        elif key == ord('r'):
            print('Setting pencil to' + Fore.RED + ' red color' + Style.RESET_ALL)
            drawing_data['color'] = (0, 0, 255)
            drawing_data['selected_color'] = (0, 0, 255)
            shapes_data['color'] = (0,0,255)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])

        elif key == ord('m'):
            var_m = 1
            blank_image = np.ones((int(height), int(width), 3), dtype=np.uint8) * 255
            cv2.setMouseCallback('White Board', click_events, (blank_image, prev_coords, drawing_data, limit))
            cv2.imshow('White Board', blank_image)      
       
        elif key == ord('g'):
            print('Setting pencil to' + Fore.GREEN + ' green color' + Style.RESET_ALL)
            drawing_data['color'] = (0, 255, 0)
            drawing_data['selected_color'] = (0, 255, 0)
            shapes_data['color'] = (0, 255, 0)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
        
        elif key == ord('b'):
            print('Setting pencil to' + Fore.BLUE + ' blue color' + Style.RESET_ALL)
            drawing_data['color'] = (255, 0, 0)
            drawing_data['selected_color'] = (255, 0, 0)
            shapes_data['color'] = (255, 0, 0)
            cv2.setTrackbarPos(trackbarInfo['trackbar_name'], color_name, trackbarInfo['val'])
        
        elif key == ord('+'):
            if drawing_data['thickness'] == 20:
                print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its maximum value.' + Style.RESET_ALL)
            else:
                drawing_data['thickness'] += 1
                shapes_data['thickness'] += 1
                print('Increasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness']))
                
        elif key == ord('-'):
            if drawing_data['thickness'] == 1:
                print(Fore.LIGHTYELLOW_EX + 'Pencil thickness is already at its minimum value.' + Style.RESET_ALL)
            else:
                drawing_data['thickness'] -= 1
                shapes_data['thickness'] -= 1
                print('Decreasing pencil thickness.\t Current thickness: ' + str(drawing_data['thickness']))    
        
        elif key == ord('c'):
            print('Clearing image')
            blank_image = np.ones((int(height),int(width),3), dtype=np.uint8)*255
        
        elif key == ord('w'):
            print('Saving image as' + Fore.GREEN + ' result.png' + Style.RESET_ALL + ' ...')
            cv2.imwrite('result.png', blank_image)
        
        elif key == ord('s')  and not shapes_data['rectangle']:
            print('Starting to draw the shape of a rectangle')
            shapes_data['rectangle'] = True
        
        elif key == ord('o') and not shapes_data['circle']:
            print('Starting to draw the shape of a circle')
            shapes_data['circle'] = True
         


    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
