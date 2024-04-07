# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:16:40 2024

@author: luisc
"""

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import math
import mediapipe as mp
import numpy as np
import cv2

########################################################################

def find_center(landmarks_eyes):
    num_points = len(landmarks_eyes)
    center_x = sum(coor[0] for coor in landmarks_eyes) / num_points
    center_y = sum(coor[1] for coor in landmarks_eyes) / num_points
    center_z = sum(coor[2] for coor in landmarks_eyes) / num_points
    return (center_x, center_y, center_z);

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result, rot_mat

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

########################################################################

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

# Landmakrs

eye_landmakrs_L = [7, 33, 246, 161, 160, 159, 158, 157, 173, 133,
                 155, 154, 153, 145, 144, 163]
eye_landmakrs_R = [362,398, 384, 385, 386, 387, 388, 466, 263, 249,
                   390, 373, 374, 380, 381, 382]

eye_landmarks_complete = eye_landmakrs_L + eye_landmakrs_R

eye_landmarks = []
eye_landmarks_CL = []
eye_landmarks_CR = []

# Create an FaceLandMaker objet
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
    

while cap.isOpened():
    
    ret, frame = cap.read()
    
    if not ret:
        break
    
    results = face_mesh.process(frame)
    
    if results.multi_face_landmarks:
        
        face_landmarks = results.multi_face_landmarks[0]
        
        landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )

        landmarks = landmarks.T
        
        landmarks = landmarks[:, :468]
        
        pose_transform_mat =
        
        cv2.imshow("Frame", frame)
        
    else:
        print("Face not found")
    
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
            
# Release resources
cap.release()
cv2.destroyAllWindows()