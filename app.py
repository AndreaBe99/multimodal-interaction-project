import argparse
import pyaudio
import time
from src.libs.recording.recorder import Recorder
from src.libs.recording.recorder_video import VideoRecorder
from src.libs.recording.recorder_audio import AudioRecorder

from src.libs.gui.controller import App


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

if __name__ == "__main__":
    # Initialize parser
    parser = argparse.ArgumentParser()
    
    # Adding optional argument
    parser.add_argument("-e", 
                        "--execution_type", 
                        choices=['CLI', 'GUI'], 
                        help = "GUI or CLI", 
                        default="GUI")
    
    parser.add_argument("-w", 
                        "--what_to_record", 
                        choices=['video', 'audio', 'both'], 
                        help = "video, audio or both", 
                        default="both")
    
    parser.add_argument("-t",
                        "--time",
                        type=int,
                        help="Time in seconds to record",
                        default=5)
    
    parser.parse_args()  
    
    if parser.parse_args().execution_type == "GUI":
        app = App()
        app.mainloop()
        
    elif parser.parse_args().execution_type == "CLI":
        video_args = {"fps":6,
                "fourcc":"MJPG",
                "device_index":0,
                "frame_counts":1,
                "frameSize":(640,480),
                "video_filename":"temp_video.avi"}
    
        audio_args = {"rate":44100,
                    "device_index":-1,
                    "frames_per_buffer":1024,
                    "py_format":pyaudio.paInt16,
                    "audio_filename":"temp_audio.wav",
                    "channel":2}
        
        recorder = Recorder(video_args=video_args, audio_args=audio_args)
        
        if parser.parse_args().what_to_record not in ['video', 'audio', 'both']:
            print("Invalid type. Valid types are: video, audio or both")
            
        elif parser.parse_args().what_to_record == "video":
            # VIDEO RECORDING
            recorder.start_video_recording()
            # Video recording for 5 seconds  by default
            time.sleep(parser.parse_args().time)
            recorder.stop_video_recording()
            
        elif parser.parse_args().what_to_record == "audio":
            # AUDIO RECORDING
            recorder.start_audio_recording()
            # Audio recording for 5 seconds  by default
            time.sleep(parser.parse_args().time)
            recorder.stop_audio_recording()
        
        elif parser.parse_args().what_to_record == "both":
            # VIDEO AND AUDIO RECORDING
            recorder.start_AVrecording()
            # Video recording for 5 seconds by default
            time.sleep(parser.parse_args().time)
            recorder.stop_AVrecording(filename="test")