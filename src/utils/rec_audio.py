import argparse
import struct
import wave
import time
import math
import threading
from pvrecorder import PvRecorder

class AudioRecorder:
    def __init__(self, device_index, output_path=None):
        self.device_index = device_index
        self.output_path = output_path
        self.recorder = None
        self.wavfile = None
        self.audio_thread = None

    def compute_decibel(self, pcm):
        """
        Calculate decibel from PCM data.

        Args:
            pcm (_type_): pcm data 

        Returns:
            decibel (float): decibel value
        """
        rms = math.sqrt(sum([(sample / 32768.0) ** 2 for sample in pcm]) / len(pcm))
        if rms > 0:
            decibel = 20 * math.log10(rms)
        else:
            decibel = -float('inf')  # Set decibel to negative infinity if rms is zero
        return decibel

    def process_audio(self):
        """
        Process audio data from recorder.
        """
        try:
            while True:
                pcm = self.recorder.read()
                decibel = self.calculate_decibel(pcm)
                if decibel > 50:
                    print("Decibel: %.2f" % decibel)
                if self.wavfile is not None:
                    self.wavfile.writeframes(struct.pack("h" * len(pcm), *pcm))
        except KeyboardInterrupt:
            print("Stopping recording thread...")

    def start_recording(self):
        """
        Start recording audio.
        """
        self.recorder = PvRecorder(device_index=self.device_index, frame_length=512)
        self.recorder.start()
        print("Using device: %s" % self.recorder.selected_device)
        print("Press Ctrl+C to stop recording...")
        try:
            if self.output_path is not None:
                self.wavfile = wave.open(self.output_path, "w")
                self.wavfile.setparams((1, 2, 16000, 512, "NONE", "NONE"))

            self.audio_thread = threading.Thread(target=self.process_audio)
            self.audio_thread.start()

            while True:
                time.sleep(1)

        except KeyboardInterrupt:
            print("Stopping main thread...")
        finally:
            self.recorder.delete()
            if self.wavfile is not None:
                self.wavfile.close()
            if self.audio_thread is not None:
                self.audio_thread.join()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--show_audio_devices",
        help="List of audio devices currently available for use.",
        action="store_true")

    parser.add_argument(
        "--audio_device_index",
        help="Index of input audio device.",
        type=int,
        default=-1)

    parser.add_argument(
        "--output_path",
        help="Path to file to store raw audio.",
        default=None)

    args = parser.parse_args()

    if args.show_audio_devices:
        devices = PvRecorder.get_audio_devices()
        for i in range(len(devices)):
            print("index: %d, device name: %s" % (i, devices[i]))
    else:
        device_index = args.audio_device_index
        output_path = args.output_path

        recorder = AudioRecorder(device_index, output_path)
        recorder.start_recording()
