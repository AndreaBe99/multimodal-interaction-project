import cv2
import time
import numpy as np

import sys
sys.path.append("./")
from src.libs.utils.gaze import Gaze
from src.libs.utils.colors import Colors

class LookingAway():
    """Class to detect if the driver is looking away"""
    def __init__(self, fps, gaze_treshold=0.2, time_treshold=2) -> None:
        """
        Args:
            fps (int): fps of the video
            gaze_treshold (float): gaze threshold
            time_treshold (float, optional): gaze time threshold. Defaults to 4.0.
        """
        self.fps = fps
        self.gaze_treshold = gaze_treshold
        self.time_treshold = time_treshold

        self.delta_time_frame = 1.0 / self.fps
        self.gaze_act_tresh = time_treshold / self.delta_time_frame
        self.gaze_counter = 0
        
        # Variable used to compute drowsiness
        # For tracking counters and sharing states in and out of callbacks.
        self.state = {
            "start_time": time.perf_counter(),
            "gaze_time": 0.0, # Holds the amount of time passed with EAR < EAR_THRESH
            "color": Colors.GREEN.value,
            "play_alarm": False,
            "gaze": 0.0,
        }
    
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
        
        gaze = Gaze()
        gaze_score = gaze.compute_gaze(frame, landmarks)
        
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
        
        # NOTE! This is an alternative way to compute the gaze time
        # from https://github.com/e-candeloro/Driver-State-Detection/tree/master
        """ 
        looking_away = False
        if self.gaze_counter >= self.gaze_act_tresh:
            looking_away = True
            self.state["color"] = Colors.RED.value
            self.state["play_alarm"] = True
            self.state["gaze"] = gaze_score
        if gaze_score is not None and gaze_score >= self.gaze_treshold:
            if not looking_away:
                self.gaze_counter += 1
        elif self.gaze_counter > 0:
            self.gaze_counter -= 1
        """
        return self.state