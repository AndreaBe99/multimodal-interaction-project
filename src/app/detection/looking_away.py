import time
import numpy as np

import sys
sys.path.append("./")
from src.app.utils.gaze import Gaze
from src.app.utils.config import Colors

class LookingAway():
    """Class to detect if the driver is looking away"""
    def __init__(self, fps, gaze_treshold=0.2, time_treshold=3) -> None:
        """
        Args:
            fps (int): fps of the video
            gaze_treshold (float): gaze threshold
            time_treshold (float, optional): gaze time threshold. Defaults to 4.0.
        """
        self.fps = fps
        self.gaze_treshold = gaze_treshold
        self.time_treshold = time_treshold
        
        # Variable used to compute drowsiness
        # For tracking counters and sharing states in and out of callbacks.
        self.state = {
            "start_time": time.perf_counter(),
            "gaze_time": 0.0, # Holds the amount of time passed with EAR < EAR_THRESH
            "color": Colors.GREEN.value,
            "play_alarm": False,
            "gaze": 0.0,
        }
        self.gaze = Gaze()
    
    def detect_looking_away(self, frame:np.array, landmarks) -> np.ndarray:
        """
        Detect if the driver is looking away using Gaze technique.
        
        Args:
            frame (np.ndarray): Frame of the video
            landmarks (np.ndarray): list of face landmarks, face mesh
        
        Returns:
            state (dict): state of the driver
        """
        if not landmarks:
            self.state["start_time"] = time.perf_counter()
            self.state["gaze_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
            return self.state
        
        gaze_score = self.gaze.compute_gaze(frame, landmarks)
        
        if gaze_score is None:
            return self.state
        
        # If the eye aspect ratio is below the blink threshold, increment the 
        # blink frame counter to detect if a person is drowsy.
        if gaze_score > self.gaze_treshold:
            # Increase DROWSY_TIME to track the time period with EAR less than 
            # the threshold and reset the start_time for the next iteration.
            end_time = time.perf_counter()
            self.state["gaze_time"] += end_time - self.state["start_time"]
            self.state["start_time"] = end_time
            self.state["color"] = Colors.RED.value
            if self.state["gaze_time"] >= self.time_treshold:
                self.state["play_alarm"] = True
        else:
            self.state["start_time"] = time.perf_counter()
            self.state["gaze_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
        
        self.state["gaze"] = gaze_score
        
        return self.state