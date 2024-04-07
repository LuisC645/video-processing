import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions

from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import os

from image_transformer import ImageTransformer
from face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis
)

########################################################################

def find_center(landmarks_eyes):
    num_points = len(landmarks_eyes)
    center_x = sum(coor[0] for coor in landmarks_eyes) / num_points
    center_y = sum(coor[1] for coor in landmarks_eyes) / num_points
    center_z = sum(coor[2] for coor in landmarks_eyes) / num_points
    return (center_x, center_y, center_z);

def roll_rotate(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, rot_mat

def export_video(array_frame, name):

    # Tamaño promedio
    size_frames = np.array([frame.shape[:2] for frame in array_frame])
    size_frame_prom = np.mean(size_frames, axis=0)
    size_frame_prom = (int(size_frame_prom[0]), int(size_frame_prom[1])) # Convertir a tupla de enteros

    resolution = size_frame_prom
    
    # Normalizar frames
    frames_normalized = [cv2.resize(frame, size_frame_prom) for frame in array_frame]
    
    # Ruta a exportar el video
    processed_folder = "processed"
    
    # Crear la carpeta si no existe
    if not os.path.exists(processed_folder):
        os.makedirs(processed_folder)
    
    # Asignar nombre
    name_processed = "processed_" + name
    
    # Ruta completa del archivo de video de salida
    path_complete = os.path.join(processed_folder, name_processed)

    # Configurar el objeto VideoWriter
    fps = 30.0
    video_processed = cv2.VideoWriter(path_complete, cv2.VideoWriter_fourcc(*'XVID'), fps, resolution)

    # Escribir los frames normalizados en el objeto VideoWriter
    for frame in frames_normalized:
        video_processed.write(frame)
        
    # Liberar el objeto VideoWriterq
    video_processed.release()
    
    print("El video se ha guardado exitosamente como:", name_processed)

def crop_image(imagen, landmarks):
    L_margin = 10
    R_margin = 10
    T_margin = 30
    B_margin = 10
    
    # Obtener las coordenadas de los landmarks T, R, L y B
    x_T, y_T = int(landmarks[10].x * imagen.shape[1]), int(landmarks[10].y * imagen.shape[0])
    x_L, y_L = int(landmarks[234].x * imagen.shape[1]), int(landmarks[234].y * imagen.shape[0])
    x_R, y_R = int(landmarks[454].x * imagen.shape[1]), int(landmarks[454].y * imagen.shape[0])
    x_B, y_B = int(landmarks[152].x * imagen.shape[1]), int(landmarks[152].y * imagen.shape[0])

    # Calcular los límites del rectángulo para el recorte
    x_min = max(0, x_L - L_margin)
    x_max = min(imagen.shape[1], x_R + R_margin)
    y_min = max(0, y_T - T_margin)
    y_max = min(imagen.shape[0], y_B + B_margin)

    # Verificar si las coordenadas de recorte son válidas
    if x_min >= x_max or y_min >= y_max:
        return None  # Devolver None si las dimensiones son inválidas
    
    # Recortar la imagen
    cropped_image = imagen[y_min:y_max, x_min:x_max]

    return cropped_image, (x_min, y_min, x_max, y_max)

def crop_frame(image, landmarks):
    # Obtener las coordenadas de los landmarks
    top_x, top_y = int(landmarks[0][0] * image.shape[1]), int(landmarks[0][1] * image.shape[0])
    left_x, left_y = int(landmarks[1][0] * image.shape[1]), int(landmarks[1][1] * image.shape[0])
    right_x, right_y = int(landmarks[2][0] * image.shape[1]), int(landmarks[2][1] * image.shape[0])
    bottom_x, bottom_y = int(landmarks[3][0] * image.shape[1]), int(landmarks[3][1] * image.shape[0])

    # Calcular los límites del rectángulo para el recorte
    x_min = max(0, left_x - 10)
    y_min = max(0, top_y - 10)
    x_max = min(image.shape[1], right_x + 10)
    y_max = min(image.shape[0], bottom_y + 10)

    # Recortar la imagen
    cropped_image = image[y_min:y_max, x_min:x_max]

    return cropped_image


########################################################################
