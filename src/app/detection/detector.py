import cv2
import typing
import numpy as np

import sys

sys.path.append("./")
from src.app.utils.config import Colors
from src.app.detection.distracted import Distracted
from src.app.detection.drowsiness import Drowsiness
from src.app.detection.looking_away import LookingAway
from src.app.detection.loudness import Loudness
from src.app.utils.tts import SimpleTextToSpeech
from src.train.config import StaticDataset as sd


class Detector:
    def __init__(self, rec="both", fps=6, height=480, width=640):
        self.width = width
        self.height = height

        self.drowsiness = Drowsiness(width=width, height=height)
        self.looking_away = LookingAway(fps=fps)
        self.loudness = Loudness()
        self.tts = SimpleTextToSpeech()
        self.distracted = Distracted()
        self.rec = rec  # video, audio, both

        # Plot text
        self.txt_origin_gaze_time = (10, int(height // 2 * 1.55))
        self.txt_origin_drowsy = (10, int(height // 2 * 1.7))
        self.txt_origin_alarm = (10, int(height // 2 * 1.85))
        self.txt_origin_ear = (10, int(height // 2 * 0.1))  # (10, 30)
        self.txt_origin_gaze = (10, int(height // 2 * 0.25))  # (10, 40)

    def detect(
        self,
        frame_counts: int = 1,
        frame: np.array = None,
        landmarks: np.array = None,
        audio_data: np.array = None,
        side_camera: int = 2,
    ) -> np.ndarray:
        """
        Detect drowsiness, looking away, and loudness

        Args:
            frame_counts (int): frame counter
            frame (np.array): frame of the video
            landmarks (np.array): landmarks of the face
            audio_data (np.array): audio data
            side_camera (int): 
                - 0 use only the side camera and only the model detection, 
                - 1 use only the front camera and only the ear and gaze detection, 
                - 2 use front camera but both predictions, model, ear and gaze detection

        Returns:
            frame (np.array): frame with landmarks and text
        """
        blink = None
        state_distraction = None
        state_drownsiness = None
        state_looking_away = None
        state_loudness = None

        if (self.rec == "video" or self.rec == "both") and landmarks is not None:
            # We want to execute the detection with the model every 3 frames
            # because it is eavy withouth GPU
            
            if side_camera == 0 or side_camera == 2:
                # if frame_counts % 3 == 0:
                # Detect distraction
                state_distraction = self.distracted.detect_distraction(frame)
            if side_camera == 1 or side_camera == 2:
                # Detect drowsiness
                state_drownsiness = self.drowsiness.detect_drowsiness(frame, landmarks)

                # Detect looking away
                state_looking_away = self.looking_away.detect_looking_away(frame, landmarks)

            # Plot text on the frame
            frame, blink = self.plot_text_on_frame(
                frame=frame,
                state_distraction=state_distraction,
                state_drownsiness=state_drownsiness,
                state_looking_away=state_looking_away,
            )

            # NOTE: TO TEST ONLY THE MODEL
            # frame, blink = self.plot_text_on_frame(frame, state_distraction)
            # NOTE: TO TEST ONLY THE OPENCV FUNCTIONALITY
            # frame, blink = self.plot_text_on_frame(frame, state_drownsiness=state_drownsiness, state_looking_away=state_looking_away)

        if (self.rec == "audio" or self.rec == "both") and audio_data is not None:
            # Detect loudness
            state_loudness = self.loudness.compute_loudness(audio_data)
            # self.loudness.display_loudness(rms)

        return frame, blink, state_loudness

    def plot_text_on_frame(
        self,
        frame: np.array,
        state_distraction: typing.Dict[str, typing.Any] = None,
        state_drownsiness: typing.Dict[str, typing.Any] = None,
        state_looking_away: typing.Dict[str, typing.Any] = None,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        fntScale=0.8,
        thickness=2,
    ) -> np.ndarray:
        """
        Plot text on the frame

        Args:
            frame (np.array): Frame of the video
            state_distraction (typing.Dict[str, typing.Any]): State of distraction
            state_drowness (typing.Dict[str, typing.Any]): State of drowsiness
            state_looking_away (typing.Dict[str, typing.Any]): State of looking away
            font (cv2.FONT, optional): Font of the text. Defaults to cv2.FONT_HERSHEY_SIMPLEX.
            fntScale (float, optional): Font scale. Defaults to 0.8.
            thickness (int, optional): Thickness of the text. Defaults to 2.

        Returns:
            frame (np.array): Frame with the text
        """
        # NOTE! The following check is a placeholder.
        # Check if the state are None, then put placeholder
        if state_drownsiness is None:
            state_drownsiness = {
                "color": Colors.GREEN.value,
                "drowsy_time": 0,
                "ear": 0,
                "play_alarm": False,
            }
        if state_looking_away is None:
            state_looking_away = {
                "color": Colors.GREEN.value,
                "gaze_time": 0,
                "gaze": 0,
                "play_alarm": False,
            }
        if state_distraction is None:
            state_distraction = {
                "color": Colors.GREEN.value,
                "play_alarm": False,
                "class": "c0",
            }
        #################################################

        blink = False
        alarm_color = Colors.GREEN.value

        txt_alarm = sd.ACTIVITY_MAP.value["c0"]

        if state_drownsiness["play_alarm"]:
            txt_alarm = "WAKE UP!"
            alarm_color = Colors.RED.value
            self.tts.speak("WAKE UP!")
            blink = True

        if state_looking_away["play_alarm"]:
            txt_alarm = "LOOK STRAIGHT!"
            alarm_color = Colors.RED.value
            self.tts.speak("LOOK STRAIGHT!")
            blink = True

        if state_distraction["play_alarm"]:
            txt_alarm = sd.ACTIVITY_MAP.value[state_distraction["class"]]
            alarm_color = Colors.RED.value
            self.tts.speak(
                "Detected: " + sd.ACTIVITY_MAP.value[state_distraction["class"]]
            )
            blink = True
        # else:
        #   txt_alarm = sd.ACTIVITY_MAP.value["c0"]

        txt_drowsy = "DROWSY TIME: {:.2f}".format(state_drownsiness["drowsy_time"])
        txt_ear = "EAR: {:.2f}".format(state_drownsiness["ear"])

        txt_looking = "GAZE TIME: {:.2f}".format(state_looking_away["gaze_time"])
        txt_gaze = "GAZE: {:.2f}".format(state_looking_away["gaze"])

        # Function to get a smaller list of parameters for cv2.putText
        def put_text(
            frame, txt, origin, color, font=font, fntScale=fntScale, thickness=thickness
        ):
            return cv2.putText(frame, txt, origin, font, fntScale, color, thickness)

        put_text(
            frame, txt_looking, self.txt_origin_gaze_time, state_looking_away["color"]
        )
        put_text(frame, txt_gaze, self.txt_origin_gaze, state_looking_away["color"])

        put_text(frame, txt_drowsy, self.txt_origin_drowsy, state_drownsiness["color"])
        put_text(frame, txt_ear, self.txt_origin_ear, state_drownsiness["color"])

        put_text(frame, txt_alarm, self.txt_origin_alarm, alarm_color)

        return frame, blink
