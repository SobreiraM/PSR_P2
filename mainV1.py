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

    parser = argparse.ArgumentParser(description='Definição do modo de teste ' )
    parser.add_argument("-j", "--json", 
                        required=True, 
                        help="Caminho completo para o arquivo JSON")
    return parser.parse_args()

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

def Filter_Desired_Areas(image,limits):

    lower_limits = (limits['min_B'], limits['min_G'], limits['min_R'])
    upper_limits = (limits['max_B'], limits['max_G'], limits['max_R'])
    mask = cv2.inRange(image, lower_limits, upper_limits)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result

def Find_Largest_Contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea, default=0)

    return largest_contour

def Draw_Largest_Contour(image, contour):
        cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)  # Fill the contour

def Process_Frame(frame, limits):
    # Filter desired areas
    filtered_frame = Filter_Desired_Areas(frame,limits)

    # Find the largest contour
    largest_contour = Find_Largest_Contour(filtered_frame)

    # Create a black frame
    result_frame = np.zeros_like(frame)

    # Draw and fill the largest contour on the black frame
    Draw_Largest_Contour(result_frame, largest_contour)

    # Mark the center of largest contour with a circle
    centroid = np.mean(np.argwhere(result_frame),axis=0)
    centroid_x, centroid_y = int(centroid[1]), int(centroid[0])

    result_frame_dot = cv2.circle(frame, (centroid_x,centroid_y), 3, (255,0,0),-1)

    return result_frame,result_frame_dot, centroid_x, centroid_y

def Drawing(drawing_data,blank_image):

    cv2.line(blank_image,(drawing_data['new_x'],drawing_data['new_y']),
              (drawing_data['previous_x'],drawing_data['previous_y']),drawing_data['color'], drawing_data['thickness'])
        
    drawing_data['previous_x'] = drawing_data['new_x']
    drawing_data['previous_y'] = drawing_data['new_y']

def main():

    args = arg_function()

    # Abrir o arquivo JSON
    with open(args.json, 'r') as arquivo:
        limits = json.load(arquivo)

    # Imprimir as informações lidas do JSON
    print("min_B", limits["min_B"])
    print("max_B", limits["max_B"])
    print("min_G", limits["min_G"])
    print("max_G", limits["max_G"])
    print("min_R", limits["min_R"])
    print("max_R", limits["max_R"])

    # Setup da captura de video
    capture = cv2.VideoCapture(0) 

    # Mensagem de segurança da leitura
    if not capture.isOpened():
        print("ERROR.")
        return

    # Ir buscar as dimensoes iguais a janela da captura
    width = int(capture.get(3))     # valor 3 corresponde à largura do vídeo
    height = int(capture.get(4))    # valor 4 corresponde à altura do vídeo

    # Criar uma imagem em branco do mesmo tamanho da imagemm da camera
    # Criação de uma matriz height por width que é multiplicada por 255 para ficar totalmente branca
    blank_image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8) 

    # Variable Initialization
    root = tk.Tk()
    window_name = 'Webcam'
    color_name = 'Color Picker'

    drawing_data = {'new_x': 0, 'new_y': 0,'previous_x': 0, 'previous_y': 0,'selected_color': (0,0,0), 'color': (0,0,255), 'thickness': 2}
    scr_w = root.winfo_screenwidth()
    scr_h = root.winfo_screenheight()
    
    # Initialization
    rgb_image = cv2.imread('RGB_Square.png', cv2.IMREAD_COLOR)
    rgb_image = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB)
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
    cv2.namedWindow(color_name, cv2.WINDOW_NORMAL)

    
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

    # Set mouse callback
    cv2.setMouseCallback(color_name, partial(chooseColor, img=rgb_image, drawing_data=drawing_data, trackbarInfo=trackbarInfo))
    

    while True:

        _, frame = capture.read()

        mask, processed_frame, centroid_x, centroid_y = Process_Frame(frame, limits)

        drawing_data['new_x'] = centroid_x
        drawing_data['new_y'] = centroid_y 

        Drawing(drawing_data, blank_image)

        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)
        cv2.imshow('White board', blank_image)
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


    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()