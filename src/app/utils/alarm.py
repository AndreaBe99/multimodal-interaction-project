from pydub import AudioSegment
from pydyb.playback import play

class Alarm:
    def play_sound(self, sound_path, duration):
        sound = AudioSegment.from_file(sound_path)
        sound = sound[:duration * 1000] # converts the duration into milliseconds
        play(sound)

    def play_beep(self, duration):
        beep = AudioSegment.from_file("path_to_beep_sound.wav") # insert the path to the file of beep sound
        beep = beep[:duration * 1000] # converts the duration into milliseconds
        play(beep)

if __name__ == "__main__":
    player = Alarm()
    player.play_sound("path_to_audio_file.wav", 5) # play the first 5 second of the file audio
    player.play_beep(2) # play a beep of 2 seconds