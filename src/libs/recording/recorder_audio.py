import pyaudio
import wave
import threading
import sys
import numpy as np
import audioop

sys.path.append('./')
from src.libs.detection.loudness import Loudness
from src.libs.utils.config import Path


class AudioRecorder():
    # Audio class based on pyAudio and Wave
    def __init__(self, 
                 rate=44100,
                 device_index=-1, 
                 frames_per_buffer=1024, 
                 py_format=pyaudio.paInt16,  
                 audio_filename=Path.AUDIO_FILE_NAME.value,
                 channel=1 if sys.platform == 'darwin' else 2):
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
        self.stream = self.audio.open(format=self.format,
                                      channels=self.channels,
                                      rate=self.rate,
                                      input=True,
                                      frames_per_buffer = self.frames_per_buffer)
        self.audio_frames = []
    
    
    def record(self)->None:
        """
        Audio starts being recorded from the microphone and print the rms level 
        if it is greater than 0.5.
        """
        self.stream.start_stream()
        
        # Instantiate the loudness class
        loudness = Loudness()
        while(self.open == True):
            data = self.stream.read(self.frames_per_buffer) 
            self.audio_frames.append(data)
            
            # Compute the loudness of the audio and display it if it is greater 
            # than 0.5
            loudness.display_loudness(data)
            
            if self.open==False:
                break
    
    
    def stop(self)->None:
        """
        Finishes the audio recording therefore the thread too  
        """
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
    
    
    def start(self)->None:
        """
        Launches the audio recording function using a thread
        """
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()