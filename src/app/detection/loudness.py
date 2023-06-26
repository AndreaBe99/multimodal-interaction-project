import wave
import sys
import pyaudio
import audioop
import numpy as np
import time

sys.path.append('./')
from src.app.utils.config import Colors

class Loudness():
    """
    Compute the RMS level of the audio data.
    """
    
    def __init__(
        self, 
        time_treshold=1.0, 
        # rms_treshold=0.8, 
        db_treshold=60,
        audio_width=2, 
        normalization=32767) -> None:
        """
        Args:
            width (int): Width of the audio data. Default is 2.
            normalization (int): Normalization of the audio data. 
                Default is 32767.
                The division by 32767 is performed in the code to normalize the 
                RMS (Root Mean Square) value of the audio data.
                In audio processing, the values of audio samples are typically 
                represented as signed 16-bit integers, with a range from -32768 
                to 32767.
        """
        self.time_treshold = time_treshold
        self.db_treshold = db_treshold
        self.audio_width = audio_width
        self.normalization = normalization
        self.state = {
            "start_time": time.perf_counter(),
            "distracted_time": 0.0,  # Holds the amount of time passed with EAR < EAR_THRESH
            "color": Colors.GREEN.value,
            "play_alarm": False,
            "db": 0.0,
        }
        
    
    
    def compute_loudness(self, data):
        """
        Compute the rms level of the audio data.
        Return the root-mean-square of the fragment, i.e. sqrt(sum(S_i^2)/n).
        This is a measure of the power in an audio signal.
        
        Args:
            data (bytes): Audio data.
        
        Returns:
            rms (float): Root Mean Square (RMS) level of the audio data.
                Value is between 0 and 1.
        """
        data = np.frombuffer(data, dtype=np.int16)
        data = np.amax(data)
        rms = audioop.rms(data, self.audio_width) / self.normalization
        # Multiply per 20 beacuse it is a root power quantity
        db = 20 * np.log10(rms) + 120
        
        # if rms >= self.rms_treshold:
        if db >= self.db_treshold:
            end_time = time.perf_counter()
            self.state["distracted_time"] += end_time - self.state["start_time"]
            self.state["start_time"] = end_time
            self.state["color"] = Colors.RED.value
            if self.state["distracted_time"] >= self.time_treshold:
                self.state["play_alarm"] = True
        else:
            self.state["start_time"] = time.perf_counter()
            self.state["distracted_time"] = 0.0
            self.state["color"] = Colors.GREEN.value
            self.state["play_alarm"] = False
        
        self.state["db"] = db
        return self.state
    
    
    def display_loudness(self, rms)->None:
        """
        Print the rms level of the audio data.
        # NOTE! In the future this methodwill used to give a visual feedback
        
        Args:
            data (bytes): Audio data.
        """
        if rms >= 0.5:
            print('Attention!!! There is a lot of noise, the RMS value is %.3f' % rms)
        pass


if __name__ == "__main__":
    # Instantiate the loudness class
    loudness = Loudness()
    
    audio = pyaudio.PyAudio()
    chunk = 1024
    py_format = pyaudio.paInt16
    rate = 44100
    channel = 1 if sys.platform == 'darwin' else 2
    rate = 44100
    
    stream = audio.open(format=py_format, 
                        channels=channel, 
                        rate=rate, 
                        input=True)
    
    output_path = "src/data/audio/temp_audio.wav"
    print('Recording...')
    with wave.open(output_path, 'wb') as wf:
        wf.setnchannels(channel)
        wf.setsampwidth(audio.get_sample_size(py_format))
        wf.setframerate(rate)

        try:
            while True:
                data = stream.read(chunk)
                wf.writeframes(data)

                loudness.display_loudness(data)
        except KeyboardInterrupt:
            print('Done')
        except Exception as e:
            print(e)
        finally:
            stream.close()
            audio.terminate()