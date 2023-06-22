import pyaudio
import wave
import threading
import sys
import numpy as np
import audioop

sys.path.append('./')
from src.app.detection.loudness import Loudness
from src.app.utils.config import Path
from src.app.detection.detector import Detector


class AudioRecorder():
    # Audio class based on pyAudio and Wave
    def __init__(
        self, 
        rate=44100,
        device_index=-1, 
        frames_per_buffer=1024, 
        py_format=pyaudio.paInt16,  
        audio_filename=Path.AUDIO_FILE_NAME.value,
        channel=1 if sys.platform == 'darwin' else 2,
        audio_loudness_label=None,
        audio_rcs_label=None):
        """
        Args:
            rate (int): Equivalent to Human Hearing at 40 kHz.
            device_index (int): -1: Default device.
            frames_per_buffer (int): Samples: 1024,  512, 256, 128.
            py_format (int): Format of the audio data.
            audio_filename (str): Output file path.
            channel (int): Number of channels.
        """
        self.open = True
        self.rate = rate
        self.device_index = device_index
        self.frames_per_buffer = frames_per_buffer
        self.format = py_format
        
        self.path = Path.PATH_AUDIO_RECORDING.value
        self.audio_filename = self.path + audio_filename
        
        self.channels = channel
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(
            format=self.format,
            channels=self.channels,
            rate=self.rate,
            input=True,
            frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
        self.detector = Detector(rec="audio")
        self.audio_loudness_label = audio_loudness_label
        self.audio_rcs_label = audio_rcs_label

    
    
    def record(self)->None:
        """
        Audio starts being recorded from the microphone and print the rms level 
        if it is greater than 0.5.
        """
        self.stream.start_stream()
        try:
            while(self.open == True):
                data = self.stream.read(self.frames_per_buffer) 
                self.audio_frames.append(data)
                _, _, state_loudness = self.detector.detect(audio_data=data)
                
                if self.audio_loudness_label is not None:
                    self.audio_rcs_label.configure(
                        text=Path.TEXT_GUI_AUDIO_4.value + str(state_loudness["rms"]))
                if state_loudness["play_alarm"] and self.audio_loudness_label is not None:
                    self.audio_loudness_label.configure(
                        text=Path.TEXT_GUI_AUDIO_5.value +
                        str(state_loudness["rms"]),
                        text_color="red")
                
                if self.open==False:
                    break
        except Exception as e:
            print(e)
            pass
    
    
    def stop(self)->None:
        """
        Finishes the audio recording therefore the thread too  
        """
        try:
            if self.open==True:
                self.open = False
                self.stream.stop_stream()
                self.stream.close()
                self.audio.terminate()

                waveFile = wave.open(self.audio_filename, 'wb')
                waveFile.setnchannels(self.channels)
                waveFile.setsampwidth(self.audio.get_sample_size(self.format))
                waveFile.setframerate(self.rate)
                waveFile.writeframes(b''.join(self.audio_frames))
                waveFile.close()
            pass
        except Exception as e:
            print(e)
            pass
        
    
    def start(self)->None:
        """
        Launches the audio recording function using a thread
        """
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()