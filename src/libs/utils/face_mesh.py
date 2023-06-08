import cv2
import mediapipe as mp
import numpy as np

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates

    
class FaceMesh():
    """Class for face landmarks detection"""
    def __init__(
        self,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        ) -> None:
        """
        Args:
            max_num_faces (int, optional): Maximum number of faces to detect. 
                Defaults to 1.
            refine_landmarks (bool, optional): Whether to refine landmarks. 
                Defaults to True.
            min_detection_confidence (float, optional): Minimum confidence 
                value.  Defaults to 0.5.
            min_tracking_confidence (float, optional): Minimum tracking 
                confidence value. Defaults to 0.5.
        """        
        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_drawing_styles = mp.solutions.drawing_styles #helps us add styles onto the face
        self.mp_face_mesh = mp.solutions.face_mesh #main model import
        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = self.mp_face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
    
    def compute_face_landmarks(self, frame: np.ndarray) -> np.ndarray:
        """
        Compute face landmarks using Mediapipe FaceMesh solution pipeline.

        Args:
            frame (np.ndarray): Frame of the video
        Returns:
            np.ndarray: list of face landmarks
        """
        results = self.facemesh_model.process(frame)
        # Extract landmarks from the results
        if results.multi_face_landmarks:
            return results.multi_face_landmarks
        return None
    
    def plot_face_mesh(self, 
        frame: np.ndarray, 
        face_landmarks: np.ndarray)->None:
        """
        Plot face landmarks on the frame.

        Args:
            frame (np.ndarray): frame of the video
            landmarks (np.ndarray): list of face landmarks

        Returns:
            frame (np.ndarray): frame with landmarks
        """
        if face_landmarks is None:
            return
        
        for face_landmarks in face_landmarks:
            # print('face_landmarks:', face_landmarks)
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_tesselation_style())
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_CONTOURS,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_contours_style())
            self.mp_drawing.draw_landmarks(
                image=frame,
                landmark_list=face_landmarks,
                connections=self.mp_face_mesh.FACEMESH_IRISES,
                landmark_drawing_spec=None,
                connection_drawing_spec=self.mp_drawing_styles
                .get_default_face_mesh_iris_connections_style())
        #cv2.imshow('MediaPipe FaceMesh', frame)