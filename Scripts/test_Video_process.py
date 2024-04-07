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

limiter_landmarks = [10, 454, 234, 152]

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

mp_face_mesh = solutions.face_mesh
    
with mp_face_mesh.FaceMesh(
        max_num_faces = 1,
        refine_landmarks = True,
        min_detection_confidence = 0.5,
        min_tracking_confidence = 0.5
        ) as face_mesh:
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            
            for face_landmarks in results.multi_face_landmarks:
                
                # Agregar landmarks totales
                for landmark_index in eye_landmarks_complete:
                    landmark = face_landmarks.landmark[landmark_index]
                    temp_lnd = [landmark.x, landmark.y, landmark.z]
                    eye_landmarks.append(temp_lnd)
                        
                # Agregar landmarks R para encontrar el centro
                for landmark_index in eye_landmakrs_R:
                    landmark_R = face_landmarks.landmark[landmark_index]
                    temp_lnd_R = [landmark_R.x, landmark_R.y, landmark_R.z]
                    eye_landmarks_CR.append(temp_lnd_R)
   
                # Agregar landmarks L para encontrar el centro
                for landmark_index in eye_landmakrs_L:
                    landmark_L = face_landmarks.landmark[landmark_index]
                    temp_lnd_L = [landmark_L.x, landmark_L.y, landmark_L.z]
                    eye_landmarks_CL.append(temp_lnd_L)
                    
                # print("Tipo de datos de eye_landmarks:", type(eye_landmarks))
                # print("Tipo de datos de face_landmarks: ", type(face_landmarks))
                # print("Contenido de eye_landmarks:", eye_landmarks)           

                # Centro ojo L ---------------------------------------------------------------------------
                cL_x, cL_y, cL_z = find_center(eye_landmarks_CL)
                
                # Proyectar las coordenadas 3D en la imagen
                x_proj_L = int(cL_x * frame.shape[1])
                y_proj_L = int(cL_y * frame.shape[0])
                center_L = (x_proj_L, y_proj_L)
                
                # Graficar centro L
                cv2.circle(frame, (x_proj_L, y_proj_L), radius = 3, color = (0, 0, 255), thickness= -1)
                
                # Centro ojo R ---------------------------------------------------------------------------
                cR_x, cR_y, cR_z = find_center(eye_landmarks_CR)
                
                # Proyecter las coordenadas en 3D en la imagen
                x_proj_R = int(cR_x * frame.shape[1])
                y_proj_R = int(cR_y * frame.shape[0])
                center_R = (x_proj_R, y_proj_R)
                
                # Graficar centro R 
                cv2.circle(frame, (x_proj_R, y_proj_R), radius = 3, color = (0, 0, 255), thickness= -1)
                
                # Linea entre los puntos
                cv2.line(frame, center_R, center_L, (0,0,255), 2)

                # Encontrar angulo  (arctan(y/x))
                angle = math.atan2(center_L[1] - center_R[1], center_L[0] - center_R[0]) * 180 / np.pi
                
                if angle < 0:
                    angle_h = (angle + 180) * -1
                    print(angle_h)
                
                else: 
                    angle_h = 180 - angle
                    print(angle_h)

                # Crear un objeto NormalizedLandmarkList
                landmark_list = landmark_pb2.NormalizedLandmarkList()
                
                # Agregar solo los landmarks específicos a la lista
                for landmark_tuple in eye_landmarks:
                    landmark = landmark_list.landmark.add()
                    landmark.x = landmark_tuple[0]
                    landmark.y = landmark_tuple[1]
                    landmark.z = landmark_tuple[2]
                    
                # Rotar imagen (Roll)
                rotated_frame, rot_mat = rotate_image(frame, angle)
                
                # Reflejar imagen
                vertical_flip_rotated_frame = cv2.flip(rotated_frame, 0)
                final_frame = cv2.flip(rotated_frame, -1)
                
                # Dibujar los landmarks específicos
                mp_drawing.draw_landmarks(
                    image=rotated_frame,
                    landmark_list=landmark_list,
                    connections=None,
                    landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),  # Estilo de dibujo
                    connection_drawing_spec=None
                )
                
                cropped_frame, (x_min, y_min, x_max, y_max) = crop_image(rotated_frame, face_landmarks.landmark)

                cv2.rectangle(cropped_frame, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)


                # Mostrar el marco recortado
                cv2.imshow('Face Landmarks', cv2.flip(cropped_frame, 0))
                
                # Borrar landmarks del frame para mostrar los nuevos
                eye_landmarks = []
                eye_landmarks_CL = []
                eye_landmarks_CR = []
        
        else:
            print("Face not found")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# Release resources
cap.release()
cv2.destroyAllWindows()