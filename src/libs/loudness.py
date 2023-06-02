import wave
import sys
import pyaudio
import audioop
import numpy as np


class AudioRecorder:
    """
    Record audio from the microphone and compute the rms level.
    """
    def __init__(self, 
                 chunk=1024, 
                 rate=44100,
                 device_index=-1, 
                 output_path='output.wav',
                 py_format=pyaudio.paInt16,  
                 channel= 1 if sys.platform == 'darwin' else 2):
        
        # Samples: 1024,  512, 256, 128
        self.chunk = chunk
        # Equivalent to Human Hearing at 40 kHz
        self.rate = rate
        # -1: Default device
        self.device_index = device_index
        # Output file path
        self.output_path = output_path
        # Format of the audio data
        self.py_format = py_format
        # Number of channels
        self.channel = channel
        
        self.audio = pyaudio.PyAudio()
        self.stream = self.audio.open(format=self.py_format, 
                                      channels=self.channel, 
                                      rate=self.rate, 
                                      input=True)


    def compute_loudness(self, data, width=2)->float:
        """
        Compute the rms level of the audio data.
        
        Args:
            data (bytes): Audio data.
            width (int): Width of the audio data.
        
        Returns:
            rms (float): Root Mean Square (RMS) level of the audio data.
                Value is between 0 and 1.
        """
        data = np.frombuffer(data, dtype=np.int16)
        data = np.amax(data)
        return audioop.rms(data, width) / 32767


    def record_audio(self)->None:
        """
        Record audio from the microphone and print the rms level if it is
        greater than 0.5.
        """
        print('Recording...')
        with wave.open(self.output_path, 'wb') as wf:
            wf.setnchannels(self.channel)
            wf.setsampwidth(self.audio.get_sample_size(self.py_format))
            wf.setframerate(self.rate)
            
            try:
                while True:
                    data = self.stream.read(self.chunk)
                    wf.writeframes(data)
                    
                    rms = self.compute_loudness(data)
                    if rms >= 0.5:
                        print('Attention!!! There is a lot of noise, the RMS value is %.3f' % rms)
            except KeyboardInterrupt:
                print('Done')
            except Exception as e:
                print(e)
            finally:
                self.stream.close()
                self.audio.terminate()
            


if __name__ == '__main__':
    recorder = AudioRecorder()
    recorder.record_audio()