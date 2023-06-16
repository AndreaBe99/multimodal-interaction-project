from enum import Enum


class Colors(Enum):
    """
    Colors used in the GUI
    """
    RED = (0, 0, 255)  # BGR
    GREEN = (0, 255, 0)  # BGR
    BLUE = (255, 0, 0)  # BGR
    YELLOW = (0, 255, 255)  # BGR
    WHITE = (255, 255, 255)  # BGR
    BLACK = (0, 0, 0)  # BGR
    
    
class Path(Enum):
    """
    Paths and Strings used in the GUI
    """
    IMAGE_GUI_MAIN = "src/data/images/static/driver.png"
    IMAGE_GUI_VIDEO = "src/data/images/static/video-conference.png"
    IMAGE_GUI_AUDIO = "src/data/images/static/microphone.png"
    
    TEXT_GUI_MAIN_1 = "Choose the recording mode:"
    TEXT_GUI_MAIN_2 = "Audio and Video"
    TEXT_GUI_MAIN_3 = "Video Only"
    TEXT_GUI_MAIN_4 = "Audio Only"
    
    TEXT_GUI_1 = "Start"
    TEXT_GUI_2 = "Stop"
    TEXT_GUI_3 = "Go Back to Main Page"
    
    TEXT_GUI_AV_1 = "Recording Audio and Video"
    
    TEXT_GUI_AUDIO_0 = "Recording Audio"
    TEXT_GUI_AUDIO_1 = "No Volume Recorded"
    TEXT_GUI_AUDIO_2 = "Loudness: "
    TEXT_GUI_AUDIO_3 = "Normal Volume"
    TEXT_GUI_AUDIO_4 = "RCS: "
    TEXT_GUI_AUDIO_5 = "Attention!!! Loud Noise: "
    
    TEXT_GUI_VIDEO_0 = "Recording Video"
    
    AUDIO_FILE_NAME = "temp_audio.wav"
    PATH_AUDIO_RECORDING = "src/data/audio/"
    
    VIDEO_FILE_NAME = "temp_video.avi"
    PATH_VIDEO_RECORDING = "src/data/video/"