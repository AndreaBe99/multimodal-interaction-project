import argparse
import pyaudio
import time
from src.app.recording.recorder import Recorder
from src.app.recording.recorder_video import VideoRecorder
from src.app.recording.recorder_audio import AudioRecorder

from src.app.gui.controller import App


"""
##### HOW TO USE #####
From the terminal, run the following command:
    python app.py -e CLI -w video -t 10
        - to record only the video for 10 seconds 
    
    python app.py -e CLI -w audio -t 10
        - to record only the audio for 10 seconds
    
    python app.py -e CLI -w both -t 10
        - to record both audio and video for 10 seconds
        
    python app.py -e GUI: to run the GUI
        - to use the graphical interface to record audio and video
"""

class ArgParser():
    
    def __init__(self, video_args, audio_args):
        # Initialize parser
        self.parser = argparse.ArgumentParser()
        # Adding optional argument
        self.parser.add_argument("-e",
                            "--execution_type",
                            choices=['CLI', 'GUI'],
                            help="GUI or CLI",
                            default="GUI")
        self.parser.add_argument("-w",
                            "--what_to_record",
                            choices=['video', 'audio', 'both'],
                            help="video, audio or both",
                            default="both")
        self.parser.add_argument("-t",
                            "--time",
                            type=int,
                            help="Time in seconds to record",
                            default=5)
        
        self.video_args = video_args
        self.audio_args = audio_args
        
        self.app = App()
        self.recorder = Recorder(video_args=self.video_args, audio_args=self.audio_args)


    def parse_args(self):
        self.parser.parse_args()
        
        if self.parser.parse_args().execution_type == "GUI":
            self.app.mainloop()

        elif self.parser.parse_args().execution_type == "CLI":
            if self.parser.parse_args().what_to_record not in ['video', 'audio', 'both']:
                print("Invalid type. Valid types are: video, audio or both")

            elif self.parser.parse_args().what_to_record == "video":
                # VIDEO RECORDING
                self.recorder.start_video_recording()
                # Video recording for 5 seconds  by default
                time.sleep(parser.parse_args().time)
                self.recorder.stop_video_recording()

            elif self.parser.parse_args().what_to_record == "audio":
                # AUDIO RECORDING
                self.recorder.start_audio_recording()
                # Audio recording for 5 seconds  by default
                time.sleep(parser.parse_args().time)
                self.recorder.stop_audio_recording()

            elif parser.parse_args().what_to_record == "both":
                # VIDEO AND AUDIO RECORDING
                self.recorder.start_AVrecording()
                # Video recording for 5 seconds by default
                time.sleep(parser.parse_args().time)
                self.recorder.stop_AVrecording(filename="test")

if __name__ == "__main__":
    
    video_args = {"fps": 6,
                "fourcc": "MJPG",
                "device_index": 0,
                "frame_counts": 1,
                "frameSize": (640, 480),
                "video_filename": "temp_video.avi"}

    audio_args = {"rate": 44100,
                    "device_index": -1,
                    "frames_per_buffer": 1024,
                    "py_format": pyaudio.paInt16,
                    "audio_filename": "temp_audio.wav",
                    "channel": 2}
        
    parser = ArgParser(video_args=video_args, audio_args=audio_args)
    parser.parse_args()