import cv2
import time
import typing
import enum
import numpy as np
import mediapipe as mp

import sys
sys.path.append("./")

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from src.libs.utils.ear import EyeAspectRatio

class Colors(enum.Enum):
    RED = (0, 0, 255)  # BGR
    GREEN = (0, 255, 0)  # BGR
    BLUE = (255, 0, 0)  # BGR
    YELLOW = (0, 255, 255)  # BGR
    WHITE = (255, 255, 255)  # BGR
    BLACK = (0, 0, 0)  # BGR
    
class FaceLandmarks:
    """Class for face landmarks detection"""
    def __init__(
        self,
        ear_treshold: float = 0.1,
        time_treshold: float = 0.2,
        max_num_faces: int = 1,
        refine_landmarks: bool = True,
        min_detection_confidence: float = 0.5,
        min_tracking_confidence: float = 0.5,
        ) -> None:
        """
        Args:
            ear_treshold (float, optional): EAR threshold. Defaults to 0.1.
            time_treshold (float, optional): Time threshold. Defaults to 0.2.
            max_num_faces (int, optional): Maximum number of faces to detect. 
                Defaults to 1.
            refine_landmarks (bool, optional): Whether to refine landmarks. 
                Defaults to True.
            min_detection_confidence (float, optional): Minimum confidence 
                value.  Defaults to 0.5.
            min_tracking_confidence (float, optional): Minimum tracking 
                confidence value. Defaults to 0.5.
        """        
        # Initializing Mediapipe FaceMesh solution pipeline
        self.facemesh_model = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=max_num_faces,
            refine_landmarks=refine_landmarks,
            min_detection_confidence=min_detection_confidence,
            min_tracking_confidence=min_tracking_confidence,
        )
        
        self.ear_treshold = ear_treshold
        self.time_treshold = time_treshold
        
        # Variable used to compute drowsiness
        # For tracking counters and sharing states in and out of callbacks.
        self.state = {
            "start_time": time.perf_counter(),
            "drowsy_time": 0.0, # Holds the amount of time passed with EAR < EAR_THRESH
            "color": Colors.GREEN.value,
            "play_alarm": False,
            "ear": 0.0,
        }
    
    
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
            return results.multi_face_landmarks[0].landmark
        return None
    
    
    def detect_drowsiness(self, frame:np.array) -> np.ndarray:
        """
        Detect drowsiness using EAR technique, plot landmarks on the frame, and 
        if drowsiness is detected, set a variable to play an alarm sound. 
        
        Args:
            frame (np.ndarray): Frame of the video
        Returns:
            frame (np.ndarray): frame with landmarks
        """
        frame.flags.writeable = False
        frame_height, frame_width, _ = frame.shape
        
        # Compute face landmarks
        landmark = self.compute_face_landmarks(frame=frame)
        
        if not landmark:
            self.state["start_time"] = time.perf_counter()
            self.state["drowsy_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
            # Flip the frame horizontally for a selfie-view display.
            frame = cv2.flip(frame, 1)
            return frame
        
        # Compute eye aspect ratio for left and right eye
        EAR = EyeAspectRatio(landmark, frame_width, frame_height)
        ear, eye_coordinates = EAR.calculate_avg_ear()
        
        # NOTE: The EAR is a float value between 0.0 and 1.0, where 0.0 
        # indicates that the eye is completely closed and 1.0 indicates
        # that the eye is fully open.
        
        frame = self.plot_landmarks(frame, eye_coordinates)
        
        # If the eye aspect ratio is below the blink threshold, increment the 
        # blink frame counter to detect if a person is drowsy.
        if ear < self.ear_treshold:
            # Increase DROWSY_TIME to track the time period with EAR less than 
            # the threshold and reset the start_time for the next iteration.
            end_time = time.perf_counter()
            self.state["drowsy_time"] += end_time - self.state["start_time"]
            self.state["start_time"] = end_time
            self.state["color"] = Colors.RED.value
            if self.state["drowsy_time"] >= self.time_treshold:
                self.state["play_alarm"] = True
        else:
            self.state["start_time"] = time.perf_counter()
            self.state["drowsy_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
        
        self.state["ear"] = ear
        
        return frame
    
    
    def plot_landmarks(self, frame: np.ndarray, eye_coordinates, 
                       color: typing.Tuple[int, int, int] = (0, 255, 0)) -> np.ndarray:
        """
        Plot landmarks on the frame.
        
        Args:
            frame (np.ndarray): Frame of the video
            eye_coordinates (typing.Tuple[np.ndarray, np.ndarray]): list of face landmarks
            color (typing.Tuple[int, int, int], optional): Color of the landmarks. Defaults to (0, 255, 0).
            
        Returns:
            np.ndarray: Frame with landmarks
        """
        for lm_coordinates in [eye_coordinates[0], eye_coordinates[1]]:
            if lm_coordinates:
                for coord in lm_coordinates:
                    cv2.circle(frame, coord, 2, color, -1)
            
        frame = cv2.flip(frame, 1)
        return frame
    
    
    def plot_text(self, frame, font=cv2.FONT_HERSHEY_SIMPLEX, fntScale=0.8, 
                  thickness=2):
        """
        Plot text on the frame
        
        Args:
            frame (np.ndarray): image frame
            state (dict): dictionary with state variables
            font (_type_, optional): Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            fntScale (float, optional): Defaults to 0.8.
            thickness (int, optional): Defaults to 2.
            
        Returns:
            frame: image with text
        """
        frame_height, _, _ = frame.shape
        # Set the position of the displayed text
        txt_origin_drowsy = (10, int(frame_height // 2 * 1.7))
        txt_origin_alarm = (10, int(frame_height // 2 * 1.85))
        txt_origin_ear = (10, 30)
        
        if self.state["play_alarm"]:
            txt_alarm = "ALARM ON"
        else:
            txt_alarm = "ALARM OFF"
        
        txt_drowsy = "DROWSY TIME: {:.2f}".format(self.state["drowsy_time"])
        txt_ear = "EAR: {:.2f}".format(self.state["ear"])
        
        # Function to get a smaller list of parameters for cv2.putText
        def put_text(frame, txt, origin, color=self.state["color"], 
                     font=font, fntScale=fntScale, thickness=thickness):
            return cv2.putText(frame, txt, origin, 
                               font, fntScale, color, thickness)
        
        put_text(frame, txt_drowsy, txt_origin_drowsy)
        put_text(frame, txt_ear, txt_origin_ear)
        put_text(frame, txt_alarm, txt_origin_alarm)
        
        return frame


if __name__ == "__main__":
    # define a video capture object
    vid = cv2.VideoCapture(0)
    
    fl = FaceLandmarks()
    
    while (True):
        
        # Capture the video frame by frame
        ret, frame = vid.read()
        
        frame = fl.detect_drowsiness(frame)
        
        frame = fl.plot_text(frame)
        
        # Display the resulting frame
        cv2.imshow('frame', frame)
        
        # the 'q' button is set as the quitting button
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # After the loop release the cap object
    vid.release()
    # Destroy all the windows
    cv2.destroyAllWindows()