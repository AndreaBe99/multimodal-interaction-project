import customtkinter as ctk
import sys 
import threading

sys.path.append('./')
from PIL import Image

import sys 
sys.path.append('./')
from src.libs.recording.recorder_audio import AudioRecorder
from src.libs.detection.loudness import Loudness
from src.libs.utils.config import Path

class AudioFrame(ctk.CTkFrame):
    """Third Page of the GUI"""
    def __init__(self, parent, controller):
        """
        Initialize the third page
        
        Args:
            parent (tk.Frame): Parent frame
            controller (App): Controller class for the GUI
        """
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        
        self.audio_capture = None
        
        self.audio_label = ctk.CTkLabel(self, text=Path.TEXT_GUI_AUDIO_0.value)
        self.audio_label.pack(pady=20, padx=10)
        
        icon = ctk.CTkImage(
            light_image=Image.open(Path.IMAGE_GUI_AUDIO.value),
            dark_image=Image.open(Path.IMAGE_GUI_AUDIO.value),
            size=(200, 200)
            )
        
        self.audio_image = ctk.CTkLabel(self, image=icon, text="")
        self.audio_image.pack(pady=15, padx=10)
        #self.audio_image.place(relx=0.5, rely=0.40, anchor='center')ù
        
        self.audio_loudness_label = ctk.CTkLabel(
            self,
            text=Path.TEXT_GUI_AUDIO_1.value, 
            font=("Helvetica", 16), 
            text_color="yellow")
        self.audio_loudness_label.pack(pady=15, padx=10)
        
        self.audio_rcs_label = ctk.CTkLabel(self, text=Path.TEXT_GUI_AUDIO_2.value + str(0))
        self.audio_rcs_label.pack(pady=15, padx=10)
        
        
        
        self.audio_button_start = ctk.CTkButton(self, text=Path.TEXT_GUI_1.value, command=self.audio_start)
        self.audio_button_start.pack(side=ctk.LEFT, pady=12, padx=10)
        self.audio_button_start.place(relx=0.35, rely=0.87, anchor='center')
        
        self.audio_button_stop = ctk.CTkButton(self, text=Path.TEXT_GUI_2.value, command=self.audio_stop)
        self.audio_button_stop.pack(side=ctk.LEFT, pady=12, padx=10)
        self.audio_button_stop.place(relx=0.65, rely=0.87, anchor='center')
        self.audio_button_stop.configure(state=ctk.DISABLED)
        
        self.audio_button_back = ctk.CTkButton(self, text=Path.TEXT_GUI_3.value, command=self.audio_go_back)
        self.audio_button_back.pack(side=ctk.BOTTOM, pady=12, padx=10)
        # self.audio_button_back.place(relx=0.5, rely=0.90, anchor='center')
    
    def audio_start(self):
        """
        Function called when the "Start" button is pressed
        """
        self.audio_button_start.configure(state=ctk.DISABLED)
        self.audio_button_stop.configure(state=ctk.NORMAL)
        
        audio_thread = threading.Thread(target=self.record)
        audio_thread.start()
    
    
    def record(self):
        self.audio_capture = AudioRecorder()
        loudness = Loudness()
        
        self.audio_capture.stream.start_stream()
        self.audio_loudness_label.configure(
            text=Path.TEXT_GUI_AUDIO_3.value,
            font=("Helvetica", 16), 
            text_color="green")
        
        while(self.audio_capture.open):
            try:
                data = self.audio_capture.stream.read(
                    self.audio_capture.frames_per_buffer
                ) 
                self.audio_capture.audio_frames.append(data)
                # Compute the loudness of the audio and display it if it is 
                # greater than 0.5
                rcs = loudness.compute_loudness(data)
                self.audio_rcs_label.configure(text=Path.TEXT_GUI_AUDIO_4.value + str(rcs))
                if rcs > 0.5:
                    self.audio_loudness_label.configure(
                        text=Path.TEXT_GUI_AUDIO_5.value + str(rcs), 
                        text_color="red")
                
                if self.audio_capture.open==False:
                    break
            except  Exception as e:
                print(e)
                break
    
    
    def audio_stop(self):
        """
        Function called when the "Stop" button is pressed
        """
        self.audio_loudness_label.configure(text=Path.TEXT_GUI_AUDIO_1.value, 
                                            font=("Helvetica", 16), 
                                            text_color="yellow")
        self.audio_button_start.configure(state=ctk.NORMAL)
        self.audio_button_stop.configure(state=ctk.DISABLED)
        
        if self.audio_capture is not None:
            self.audio_capture.stop()
        self.audio_capture = None
        
    
    def audio_go_back(self):
        """
        Function called when the "Go Back to Main Page" button is pressed
        """
        if self.audio_capture is not None:
            self.stop()
        self.controller.show_frame("MainFrame")