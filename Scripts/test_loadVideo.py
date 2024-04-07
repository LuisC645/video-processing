from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe import solutions
import mediapipe as mp
import numpy as np
import cv2

mp_drawing = solutions.drawing_utils
mp_drawing_styles = solutions.drawing_styles

# Create an FaceLandMaker objet
base_options = python.BaseOptions(model_asset_path='face_landmarker.task')

options = vision.FaceLandmarkerOptions(base_options=base_options,
                                       output_face_blendshapes=True,
                                       output_facial_transformation_matrixes=True,
                                       num_faces=1)

detector = vision.FaceLandmarker.create_from_options(options)

path = "videoTest.wav"

cap = cv2.VideoCapture(path)

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
                
                # Obtener los límites del rectángulo que encierra los landmarks faciales
                x_values = [landmark.x for landmark in face_landmarks.landmark]
                y_values = [landmark.y for landmark in face_landmarks.landmark]
                min_x, min_y = min(x_values), min(y_values)
                max_x, max_y = max(x_values), max(y_values)
                
                # Calcular las coordenadas del rectángulo con márgenes
                margin_left = 10
                margin_right = 10
                margin_top = 30
                margin_bottom = 12
                start_x = max(int(min_x * frame.shape[1]) - margin_left, 0)
                start_y = max(int(min_y * frame.shape[0]) - margin_top, 0)
                end_x = min(int(max_x * frame.shape[1]) + margin_right, frame.shape[1])
                end_y = min(int(max_y * frame.shape[0]) + margin_bottom, frame.shape[0])
                
                # Recortar el marco original
                cropped_frame = frame[start_y:end_y, start_x:end_x]
                
                print(face_landmarks)
                
                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_tesselation_style()
                    )
                
                mp_drawing.draw_landmarks(
                    image = frame, 
                    landmark_list = face_landmarks,
                    connections = mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec = None,
                    connection_drawing_spec = mp_drawing_styles
                    .get_default_face_mesh_contours_style()
                    )
                
                # Mostrar el marco recortado
                cv2.imshow('Face Landmarks', cv2.flip(cropped_frame, 1))
        
        else:
            print("Face not found")
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
# Release resources
cap.release()
cv2.destroyAllWindows()