from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
from mediapipe.framework.formats import landmark_pb2
import mediapipe as mp
import numpy as np
import matplotlib.pyplot as plt

import cv2
import os

def draw_landmarks_on_image(rgb_image, detection_result):
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

file_path = "image.png"

img_original = cv2.imread(file_path)

specific_landmarks = [1, 19, 25]

if img_original is not None:
    print("The image has been loaded successfully")
    
    height, width = img_original.shape[:2]
    print(f"Dimension: {height} x {width}")
    
    new_height = 600
    new_width = 450
    
    img = cv2.resize(img_original, (new_width, new_height))
    
    cv2.imwrite('image.png', img)
    
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Create an FaceLandMaker objet
    
    base_options = python.BaseOptions(model_asset_path='face_landmarker.task')
    
    options = vision.FaceLandmarkerOptions(base_options=base_options,
                                           output_face_blendshapes=True,
                                           output_facial_transformation_matrixes=True,
                                           num_faces=1)
    
    detector = vision.FaceLandmarker.create_from_options(options)
    
    image = mp.Image.create_from_file('image.png')
    
    detection_result =  detector.detect(image)
    
    for detection in detection_result.detections:
        for i, landmark in enumerate(detection.landmarks):
            if i in specific_landmarks:
                # Dibujar el landmark en la imagen
                cv2.circle(image, (int(landmark.x), int(landmark.y)), 2, (0, 255, 0), -1)
    
    #Process the detection result
    annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result, specific_landmarks)
    
    cv2.imshow("Imagen", annotated_image)
    
    
else:
    print("Error")
    
# Release resource
cv2.waitKey(0)
cv2.destroyAllWindows()
    



