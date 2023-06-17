import platform
import subprocess

class TextToSpeech:
    def speak(self, text):
        operating_system = platform.system()
        if operating_system == 'Windows':
            subprocess.call(['PowerShell', '-Command', f'(New-Object -ComObject SAPI.SpVoice).Speak("{text}")'])
        elif operating_system == 'Darwin':  # MacOS
            subprocess.call(['say', text])
        elif operating_system == 'Linux':
            subprocess.call(['espeak', text])

if __name__ == "__main__":
    tts = TextToSpeech()
    tts.speak("Hello, this is just a test")