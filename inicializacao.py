#!/usr/bin/env python
import argparse
import json
import cv2
import numpy as np

def arg_function():

    parser = argparse.ArgumentParser(description='Definição do modo de teste ' )
    parser.add_argument("-j", "--json", 
                        required=True, 
                        help="Caminho completo para o arquivo JSON")
    return parser.parse_args()

def main():

    args = arg_function()

    # Abrir o arquivo JSON
    with open(args.json, 'r') as arquivo:
        limits = json.load(arquivo)

    # Imprimir as informações lidas do JSON pa ver se ta ok
    print("min_B", limits["min_B"])
    print("max_B", limits["max_B"])
    print("min_G", limits["min_G"])
    print("max_G", limits["max_G"])
    print("min_R", limits["min_R"])
    print("max_R", limits["max_R"])

    # setup da captura de video
    capture = cv2.VideoCapture(0) 

    # Mensagem de segurança da leitura
    if not capture.isOpened():
        print("ERROR.")
        return

    # Ir buscar as dimensoes iguais a janela da captura
    width = int(capture.get(3))     # valor 3 corresponde à largura do vídeo
    height = int(capture.get(4))    # valor 4 corresponde à altura do vídeo

    # Criar uma imagem em branco do mesmo tamanho da imagemm da camera
    #criação de uma matriz height por width que é multiplicada por 255 para ficar totalmente branca
    blank_image = 255 * np.ones(shape=[height, width, 3], dtype=np.uint8) 

    while True:

        ret, frame = capture.read()

        if not ret:
            print("ERROR.")
            break

        # Mostrar as janelas resultantes
        cv2.imshow('Video', frame)
        cv2.imshow('White board', blank_image)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    capture.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()