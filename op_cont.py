import cv2
import argparse
import numpy as np
from functools import partial
import json
import colorama 
from colorama import Fore, Back, Style

def Filter_Desired_Areas(image):
    lower_red = np.array([0, 0, 100])
    upper_red = np.array([100, 100, 255])

    mask = cv2.inRange(image, lower_red, upper_red)
    result = cv2.bitwise_and(image, image, mask=mask)

    return result


def Find_Largest_Contour(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    largest_contour = max(contours, key=cv2.contourArea)

    return largest_contour


def Draw_Largest_Contour(image, contour):
    cv2.drawContours(image, [contour], 0, (0, 255, 0), -1)  # Fill the contour

def Process_Frame(frame):
    # Filter desired areas
    filtered_frame = Filter_Desired_Areas(frame)

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
              (drawing_data['previous_x'],drawing_data['previous_y']),(0,255,0), 1)
        
    drawing_data['previous_x'] = drawing_data['new_x']
    drawing_data['previous_y'] = drawing_data['new_y']


def main():
    # Initialize video capture from the default camera (usually 0)
    capture = cv2.VideoCapture(0)
    
    # Ir buscar as dimensoes iguais a janela da captura
    width = int(capture.get(3))     # valor 3 corresponde à largura do vídeo
    height = int(capture.get(4))    # valor 4 corresponde à altura do vídeo

    # Criar uma imagem em branco do mesmo tamanho da imagemm da camera
    # Rriação de uma matriz height por width que é multiplicada por 255 para ficar totalmente branca
    blank_image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8) 
    
    #Utilizado para definir coordenadas iniciais do desenho utilizando a variavel de controlo start
    drawing_data = {'new_x': 0, 'new_y': 0, 'previous_x': 0, 'previous_y': 0, 'start': 0}

    while True:
        # Read a frame from the camera
        _, frame = capture.read()

        # Process the frame and get centroid coordinates
        mask, processed_frame, centroid_x, centroid_y = Process_Frame(frame)

        drawing_data['new_x'] = centroid_x
        drawing_data['new_y'] = centroid_y 

        Drawing(drawing_data, blank_image)

        # Display the processed frame
        cv2.imshow('Processed Frame', processed_frame)
        cv2.imshow('White board', blank_image)
        cv2.imshow('Mask', mask)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close the windows
    capture.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()