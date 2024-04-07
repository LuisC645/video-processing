# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:16:40 2024

@author: luisc
"""

import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2

from moviepy.editor import VideoFileClip
import numpy as np
import cv2
import os

from functions import *
from image_transformer import ImageTransformer
from face_geometry import (
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis
)

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles

# Landmakrs
eye_landmakrs_L = [7, 33, 246, 161, 160, 159, 158, 157, 173, 133,
                 155, 154, 153, 145, 144, 163]
eye_landmakrs_R = [362,398, 384, 385, 386, 387, 388, 466, 263, 249,
                   390, 373, 374, 380, 381, 382]

eye_landmarks_complete = eye_landmakrs_L + eye_landmakrs_R

limiter_landmarks = [10, 454, 234, 152]

# Mask
# base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

# options = vision.FaceLandmarkerOptions(base_options=base_options,
#                                         output_face_blendshapes=True,
#                                         output_facial_transformation_matrixes=True,
#                                         num_faces=1)

# detector = vision.FaceLandmarker.create_from_options(options)

# Yaw Transform config
frame_height, frame_width, channels = (480, 640, 3)

# pseudo camera internals
focal_length = frame_width
center = (frame_width / 2, frame_height / 2)
camera_matrix = np.array(
    [[focal_length, 0, center[0]], [0, focal_length, center[1]], [0, 0, 1]],
    dtype="double",
)

pcf = PCF(
    near=1,
    far=10000,
    frame_height=frame_height,
    frame_width=frame_width,
    fy=camera_matrix[1, 1],
)

face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces = 1,
    refine_landmarks = True,
    min_detection_confidence = 0.5,
    min_tracking_confidence = 0.5)

drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

# Obtener videos
array_frames = []

input_folder = "originals"
output_folder = "processed"

# Obtener la lista de archivos en la carpeta de entrada
files = os.listdir(input_folder)

# Filtrar solo los archivos de video
video_files = [file for file in files]

for name in video_files:
    
    # Landmarks para graficar
    eye_landmarks = []
    
    video_path = "originals/" + name
    name_processed = "processed_" + name

    cap = cv2.VideoCapture(video_path)
     
    # cap = cv2.VideoCapture(0)
    
    while cap.isOpened():
        
        ret, frame = cap.read()
        
        if not ret:
            break
        
        results = face_mesh.process(frame)
        
        if results.multi_face_landmarks:
            
            face_landmarks = results.multi_face_landmarks[0]
            
            landmarks = np.array([(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark])
    
            landmarks = landmarks.T
            
            landmarks = landmarks[:, :468]
            
            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                    landmarks.copy(), pcf
            )
            
            # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
            pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
            mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
            mp_translation_vector = pose_transform_mat[:3, 3, None]
    
            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(pose_transform_mat[:3, :3])
            
            angle_pitch = angles[0]
            angle_yaw = angles[1]
            angle_roll = angles[2]
            
            # print(angle_yaw, angle_roll, angle_pitch)
            # angles = [Pitch, Yaw, Roll]
            
            # and (angle_pitch <= -150 or angle_pitch >= 150)
            
            if -25 <= angle_yaw <= 25:
                
                # Yaw Rotation
                it = ImageTransformer(frame, (frame.shape[1], frame.shape[0]))
                
                yaw_rotation = it.rotate_along_axis(phi = angle_yaw)
                
                # Roll Rotation
                roll_rotation, rot_mat_roll = roll_rotate(yaw_rotation, angle_roll)
                
                array_frames.append(roll_rotation)
                
                # Graficar landmarks específicos
                # for face_landmarks in results.multi_face_landmarks:
                    
                #     # Agregar landmarks totales
                #     for landmark_index in eye_landmarks_complete:
                #         landmark = face_landmarks.landmark[landmark_index]
                #         temp_lnd = [landmark.x, landmark.y, landmark.z]
                #         eye_landmarks.append(temp_lnd)
                
                #     # Crear un objeto NormalizedLandmarkList
                #     landmark_list = landmark_pb2.NormalizedLandmarkList()
                    
                #     # Agregar solo los landmarks específicos a la lista
                #     for landmark_tuple in eye_landmarks:
                #         landmark = landmark_list.landmark.add()
                #         landmark.x = landmark_tuple[0]
                #         landmark.y = landmark_tuple[1]
                #         landmark.z = landmark_tuple[2]
                        
                #     # Dibujar los landmarks específicos
                #     mp_drawing.draw_landmarks(
                #         image=roll_rotation,
                #         landmark_list=landmark_list,
                #         connections=None,
                #         landmark_drawing_spec=mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=1),  # Estilo de dibujo
                #         connection_drawing_spec=None
                #     )
                    
                #     eye_landmarks = []
                
                
                # Cropear imagen
                
                landmark = landmarks.T
                limitters = []
                
                for i in limiter_landmarks:
                    
                    lnm_x = landmark[i][0]
                    lnm_y = landmark[i][1]
                    limitters.append([lnm_x, lnm_y])
                    
                cv2.imshow("Frame", cropped_frame)
                
            else:
                print("Face not found")
            
            eye_landmakrs = []
            
        else:
            print("Face not found")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
    # Export video
    #export_video(array_frames, name_processed)
    
    # Limpiar arrays de frames para otro video
    array_frames = []
    # Release resources
    cap.release()
    cv2.destroyAllWindows()