import cv2
import typing
import numpy as np

import sys
sys.path.append("./")
from src.app.utils.config import Colors
from src.app.detection.drowsiness import Drowsiness
from src.app.detection.looking_away import LookingAway
from src.app.detection.loudness import Loudness


class Detector():
    def __init__(self, rec="both", fps=6, height=480, width=640):
        self.drowsiness = Drowsiness()
        self.looking_away = LookingAway(fps=fps)
        self.loudness = Loudness()
        self.rec = rec # video, audio, both
        
        # Plot text
        self.txt_origin_gaze_time = (10, int(height // 2 * 1.55))
        self.txt_origin_drowsy = (10, int(height // 2 * 1.7))
        self.txt_origin_alarm = (10, int(height // 2 * 1.85))
        self.txt_origin_ear = (10, int(height // 2 * 0.1)) # (10, 30)
        self.txt_origin_gaze = (10, int(height // 2 * 0.25)) # (10, 40)
        
    
    def detect(
        self, 
        frame:np.array, 
        landmarks:np.array=None, 
        audio_data:np.array=None) -> np.ndarray:
        """
        Detect drowsiness, looking away, and loudness

        Args:
            frame (np.array): frame of the video
            landmarks (np.array): landmarks of the face
            audio_data (np.array): audio data

        Returns:
            frame (np.array): frame with landmarks and text
        """
        if (self.rec == "video" or self.rec == "both") and landmarks is not None:
            # Detect drowsiness
            state_drowness = self.drowsiness.detect_drowsiness(frame, landmarks)
            # Detect looking away
            state_looking_away = self.looking_away.detect_looking_away(frame, landmarks)
            # Plot text on the frame
            self.plot_text_on_frame(frame, state_drowness, state_looking_away)
        
        if (self.rec == "audio" or self.rec == "both") and audio_data is not None:
            # Detect loudness
            rms = self.loudness.compute_loudness(audio_data)
            self.loudness.display_loudness(rms)
        
        return frame
    
    
    def plot_text_on_frame(
        self, 
        frame:np.array, 
        state_drowness: typing.Dict[str, typing.Any],
        state_looking_away: typing.Dict[str, typing.Any],
        font=cv2.FONT_HERSHEY_SIMPLEX, 
        fntScale=0.8, 
        thickness=2) -> np.ndarray:
        """
        Plot text on the frame

        Args:
            frame (np.array): Frame of the video
            state_drowness (typing.Dict[str, typing.Any]): State of drowsiness
            state_looking_away (typing.Dict[str, typing.Any]): State of looking away
            font (cv2.FONT, optional): Font of the text. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            fntScale (float, optional): Font scale. Defaults to 0.8.
            thickness (int, optional): Thickness of the text. Defaults to 2.

        Returns:
            frame (np.array): Frame with the text
        """
        alarm_color = Colors.GREEN.value
        if state_drowness["play_alarm"]:
            txt_alarm = "WAKE UP!"
            alarm_color = Colors.RED.value
        if state_looking_away["play_alarm"]:
            txt_alarm = "LOOK STRAIGHT!"
            alarm_color = Colors.RED.value
        else:
            txt_alarm = "ALARM OFF"
            
        txt_drowsy = "DROWSY TIME: {:.2f}".format(state_drowness["drowsy_time"])
        txt_ear = "EAR: {:.2f}".format(state_drowness["ear"])
        
        txt_looking = "GAZE TIME: {:.2f}".format(state_looking_away["gaze_time"])
        txt_gaze = "GAZE: {:.2f}".format(state_looking_away["gaze"])
        
        # Function to get a smaller list of parameters for cv2.putText
        def put_text(
            frame, txt, origin, color, 
            font=font, fntScale=fntScale, thickness=thickness):
            return cv2.putText(
                frame, txt, origin, font, fntScale, color, thickness)
        
        put_text(frame, txt_looking, self.txt_origin_gaze_time, 
            state_looking_away["color"])
        put_text(frame, txt_gaze, self.txt_origin_gaze,
            state_looking_away["color"])
        
        put_text(frame, txt_drowsy, self.txt_origin_drowsy,
            state_drowness["color"])
        put_text(frame, txt_ear, self.txt_origin_ear,
            state_drowness["color"])
        
        put_text(frame, txt_alarm, self.txt_origin_alarm, alarm_color)
        
        return frame