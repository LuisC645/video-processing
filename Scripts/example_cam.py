# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 12:27:08 2024

@author: luisc
"""
import cv2
import mediapipe as mp
import numpy as np

from image_transformer import ImageTransformer
from custom.face_geometry import (  # isort:skip
    PCF,
    get_metric_landmarks,
    procrustes_landmark_basis,
)

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

cap = cv2.VideoCapture(0) #depends on your system 0 or 1

mp_drawing = mp.solutions.drawing_utils
mp_face_mesh = mp.solutions.face_mesh

face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.9)
drawing_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)

while True:
    
    success, image = cap.read()
    if success == True:
        
        results = face_mesh.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
       
        if not results.multi_face_landmarks:
            print("Face not detected")
            continue
        
        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]
            landmarks = np.array(
                [(lm.x, lm.y, lm.z) for lm in face_landmarks.landmark]
            )
            # print(landmarks.shape)
            landmarks = landmarks.T
            
            landmarks = landmarks[:, :468]

            metric_landmarks, pose_transform_mat = get_metric_landmarks(
                landmarks.copy(), pcf
            )

            # see here:
            # https://github.com/google/mediapipe/issues/1379#issuecomment-752534379
            pose_transform_mat[1:3, :] = -pose_transform_mat[1:3, :]
            mp_rotation_vector, _ = cv2.Rodrigues(pose_transform_mat[:3, :3])
            mp_translation_vector = pose_transform_mat[:3, 3, None]

            angles, mtxR, mtxQ, Qx, Qy, Qz = cv2.RQDecomp3x3(pose_transform_mat[:3, :3])
                
            print(angles)

            yaw_rotation=angles[1]
            
            pitch_rotation_aux = angles[0]
            
            if(pitch_rotation_aux>0):
                pitch_rotation = 180 - pitch_rotation_aux
            else:
                pitch_rotation = -(180 + pitch_rotation_aux)
            
            
            
            # it = ImageTransformer(image, (image.shape[1], image.shape[0]))

            # rotated_img = it.rotate_along_axis(phi = yaw_rotation) #phi y theta
            

        cv2.imshow('Face Mesh Detection', image)

        # Press Q on keyboard to  exit
        if cv2.waitKey(25) & 0xFF == ord('q'):
          break
    else:
        break
          
cap.release()
cv2.destroyAllWindows()

























