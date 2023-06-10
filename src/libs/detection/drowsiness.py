import cv2
import time
import typing
import numpy as np
import mediapipe as mp

import sys
sys.path.append("./")

from mediapipe.python.solutions.drawing_utils import _normalized_to_pixel_coordinates as denormalize_coordinates
from src.libs.utils.ear import EyeAspectRatio
from src.libs.utils.colors import Colors
    
class Drowsiness():
    """Class to detect drowsiness"""
    def __init__(
        self,
        ear_treshold: float = 0.1,
        time_treshold: float = 0.2,
        ) -> None:
        """
        Args:
            ear_treshold (float, optional): EAR threshold. Defaults to 0.1.
            time_treshold (float, optional): Time threshold. Defaults to 0.2.
        """        
        
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
    
    
    def detect_drowsiness(self, frame:np.array, landmarks) -> np.ndarray:
        """
        Detect drowsiness using EAR technique, plot landmarks on the frame, and 
        if drowsiness is detected, set a variable to play an alarm sound. 
        
        Args:
            frame (np.ndarray): Frame of the video
            landmarks (np.ndarray): list of face landmarks, face mesh
        Returns:
            frame (np.ndarray): frame with landmarks
        """
        frame.flags.writeable = False
        frame_height, frame_width, _ = frame.shape
        
        
        if not landmarks:
            self.state["start_time"] = time.perf_counter()
            self.state["drowsy_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
            return frame
        
        # Compute eye aspect ratio for left and right eye
        ear = EyeAspectRatio(landmarks.landmark, frame_width, frame_height)
        ear_avg, eye_coordinates = ear.calculate_avg_ear()
        
        # NOTE: The EAR is a float value between 0.0 and 1.0, where 0.0 
        # indicates that the eye is completely closed and 1.0 indicates
        # that the eye is fully open.
        
        frame = self.plot_landmarks(frame, eye_coordinates)
        
        # If the eye aspect ratio is below the blink threshold, increment the 
        # blink frame counter to detect if a person is drowsy.
        if ear_avg < self.ear_treshold:
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
        
        self.state["ear"] = ear_avg
        
        return self.state
    
    
    def plot_landmarks(self, frame: np.ndarray, eye_coordinates, 
                       color=Colors.BLUE.value ) -> np.ndarray:
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
        return frame