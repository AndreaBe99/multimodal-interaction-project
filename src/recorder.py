import cv2
import pyaudio
import wave
import threading
import time
import subprocess
import os
import sys
sys.path.append("./")

from src.recorder_video import VideoRecorder
from src.recorder_audio import AudioRecorder

class Recorder():
    # Class to manage the recording of audio and video
    def __init__(self, video_args, audio_args):
        self.video_thread = None
        self.audio_thread = None
        self.video_args = video_args
        self.audio_args = audio_args
        
        self.path = "src/data/"
        self.path_video = self.path + "video/"
        self.video_filename = self.path_video + self.video_args["video_filename"]
        self.path_audio = self.path + "audio/"
        self.audio_filename = self.path_audio + self.audio_args["audio_filename"]
    
    
    def start_AVrecording(self)->None:
        """
        Start the recording of audio and video.
        """
        self.video_thread = VideoRecorder(**self.video_args)
        self.audio_thread = AudioRecorder(**self.audio_args)
        self.audio_thread.start()
        self.video_thread.start()


    def start_video_recording(self)->None:
        """
        Start the recording of video.
        """
        self.video_thread = VideoRecorder()
        self.video_thread.start()


    def start_audio_recording(self)->None:
        """
        Start the recording of audio.
        """
        self.audio_thread = AudioRecorder()
        self.audio_thread.start()


    def stop_AVrecording(self, filename)->None:
        """
        Stop the recording of audio and video.
        
        Args:
            filename (str): Name of the file to be saved.
        """
        self.audio_thread.stop() 
        frame_counts = self.video_thread.frame_counts
        elapsed_time = time.time() - self.video_thread.start_time
        recorded_fps = frame_counts / elapsed_time
        print("total frames " + str(frame_counts))
        print("elapsed time " + str(elapsed_time))
        print("recorded fps " + str(recorded_fps))
        self.video_thread.stop() 
        # Makes sure the threads have finished
        while threading.active_count() > 1:
            time.sleep(1)
        # Merging audio and video signal
        if abs(recorded_fps - 6) >= 0.01:
            # If the fps rate was higher/lower than expected, re-encode it to the expected    
            print("Re-encoding")
            cmd = "ffmpeg -r " + str(recorded_fps) + " -i " + \
                self.video_filename + " -pix_fmt yuv420p -r 6 " + \
                    self.path_video + "temp_video2.avi"
            subprocess.call(cmd, shell=True)
            print("Muxing")
            cmd = "ffmpeg -ac 2 -channel_layout stereo -i " + \
                self.audio_filename + " -i " + self.path_audio + \
                    "temp_video2.avi -pix_fmt yuv420p " + filename + ".avi"
            subprocess.call(cmd, shell=True)
        else:
            print("Normal recording\nMuxing")
            cmd = "ffmpeg -ac 2 -channel_layout stereo -i " + \
                self.audio_filename + " -i " + self.video_filename + \
                    " -pix_fmt yuv420p " + filename + ".avi"
            subprocess.call(cmd, shell=True)
            print("..")
    
    
    def stop_video_recording(self)->None:
        """
        Stop the recording of video.
        """
        self.video_thread.stop()
    
    
    def stop_audio_recording(self)->None:
        """
        Stop the recording of audio.
        """
        self.audio_thread.stop()
    
    
    def file_manager(self, filename)->None:
        """
        Remove the temporary files.
        
        Args:
            filename (str): Name of the file to be saved.
        """
        # NOTE! CHANGE THE PATH TO src/data/video OR src/data/audio
        local_path = os.getcwd()
        if os.path.exists(str(local_path) + "/temp_audio.wav"):
            os.remove(str(local_path) + "/temp_audio.wav")
        if os.path.exists(str(local_path) + "/temp_video.avi"):
            os.remove(str(local_path) + "/temp_video.avi")
        if os.path.exists(str(local_path) + "/temp_video2.avi"):
            os.remove(str(local_path) + "/temp_video2.avi")
        if os.path.exists(str(local_path) + "/" + filename + ".avi"):
            os.remove(str(local_path) + "/" + filename + ".avi")


if __name__ == "__main__":
    
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
    
    # VIDEO AND AUDIO RECORDING
    # recorder.start_AVrecording()
    # Video recording for 5 seconds
    # time.sleep(5)
    # recorder.stop_AVrecording(filename="test")
    
    # VIDEO RECORDING
    recorder.start_video_recording()
    # Video recording for 5 seconds
    time.sleep(5)
    recorder.stop_video_recording()
    
    # AUDIO RECORDING
    # recorder.start_audio_recording()
    # Audio recording for 5 seconds
    # time.sleep(5)
    # recorder.stop_audio_recording()