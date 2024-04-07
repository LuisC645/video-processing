# -*- coding: utf-8 -*-
"""
Created on Mon Mar 18 13:06:54 2024

@author: luisc
"""

# -*- coding: utf-8 -*-
"""
Created on Sat Mar 16 13:16:40 2024

@author: luisc
"""

from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np

import cv2

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles
  face_landmarks_list = detection_result.face_landmarks
  annotated_image = np.copy(rgb_image)

  # Loop through the detected faces to visualize.
  for idx in range(len(face_landmarks_list)):
    face_landmarks = face_landmarks_list[idx]

    # Draw the face landmarks.
    face_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
    face_landmarks_proto.landmark.extend([
      landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z) for landmark in face_landmarks
    ])

    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_tesselation_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_CONTOURS,
        landmark_drawing_spec=None,
        connection_drawing_spec=mp.solutions.drawing_styles
        .get_default_face_mesh_contours_style())
    solutions.drawing_utils.draw_landmarks(
        image=annotated_image,
        landmark_list=face_landmarks_proto,
        connections=mp.solutions.face_mesh.FACEMESH_IRISES,
          landmark_drawing_spec=None,
          connection_drawing_spec=mp.solutions.drawing_styles
          .get_default_face_mesh_iris_connections_style())

  return annotated_image

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
        
        for face_landmarks in results.multi_face_landmarks:
            print(face_landmarks)
            mp_drawing.draw_landmarks(
                image = frame, 
                landmark_list = face_landmarks,
                connections = mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec = None,
                connection_drawing_spec = mp_drawing_styles
                .get_default_face_mesh_tesselation_style()
                )

        cv2.imshow('Face Landmarks', cv2.flip(frame, 1))
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# Release resources
cap.release()
cv2.destroyAllWindows()