import argparse
import pyaudio
import time
from src.app.recording.recorder import Recorder
from src.app.recording.recorder_video import VideoRecorder
from src.app.recording.recorder_audio import AudioRecorder
from src.app.utils.config import Path
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
        self.time_default = 5
        # Initialize parser
        self.parser = argparse.ArgumentParser()
        # Adding optional argument
        self.subparsers = self.parser.add_subparsers(help="sub-command help")
        
        # Reproduce video from file
        self.from_file_parser = self.subparsers.add_parser("from_file", help="Reproduce video from file")
        self.from_file_parser.add_argument(
            "-p", "--path_video", help="Path of the video to analyze")
        
        # Record video from webcam in real time
        self.from_camera_parser = self.subparsers.add_parser("from_camera", help="Reproduce video from camera")
        self.from_camera_parser.add_argument(
            "-e", "--execution_type", choices=['CLI', 'GUI'], help="GUI or CLI",
            default="GUI", required=True)
        self.from_camera_parser.add_argument(
            "-w", "--what_to_record", choices=['video', 'audio', 'both'],
            help="video, audio or both", default="both")
        self.from_camera_parser.add_argument(
            "-t", "--time", type=int, help="Time in seconds to record", default=self.time_default)

        self.video_args = video_args
        self.audio_args = audio_args
        
        # Check if the video is from a file or from the webcam and add the path
        # to the video recorder arguments
        self.video_from_path = False
        self.args = self.parser.parse_args()

        if "path_video" in self.args:
            self.video_args["video_from_file_path"] = self.args.path_video
            self.video_from_path = True

        self.app = App()
        self.recorder = Recorder(video_args=self.video_args, audio_args=self.audio_args)


    def parse_args(self):
        self.parser.parse_args()

        if "path_video" in self.args:
            # Reproduce video from file
            print("Reproducing video...")
            self.recorder.start_video_recording(self.video_from_path)
        else:
            if "execution_type" in self.args:
                
                if self.args.execution_type == "GUI":
                    self.app.mainloop()
                
                elif self.args.execution_type == "CLI":
                    
                    if "what_to_record" in self.args:
                        if self.args.what_to_record not in ['video', 'audio', 'both']: 
                            print("Invalid type. Valid types are: video, audio or both")
                        else:
                            if "time" in self.args:
                                self.time_default = self.args.time
                                
                            if self.args.what_to_record == "video":
                                # VIDEO RECORDING
                                print("Recording video...")
                                self.recorder.start_video_recording()
                                # Video recording for 5 seconds
                                time.sleep(self.time_default)
                                self.recorder.stop_video_recording()
                            elif self.args.what_to_record == "audio":
                                # AUDIO RECORDING
                                print("Recording audio...")
                                self.recorder.start_audio_recording()
                                # Audio recording for 5 seconds  by default
                                time.sleep(self.time_default)
                                self.recorder.stop_audio_recording()
                            elif self.args.what_to_record == "both":
                                # VIDEO AND AUDIO RECORDING
                                print("Recording video and audio...")
                                self.recorder.start_AVrecording()
                                # Video recording for 5 seconds by default
                                time.sleep(self.time_default)
                                self.recorder.stop_AVrecording(filename="test")

if __name__ == "__main__":
    
    video_args = {"fps": 15,
                "fourcc": "MJPG",
                "device_index": -1,
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