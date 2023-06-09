import customtkinter as ctk
from PIL import Image

import sys 
sys.path.append('./')
from src.libs.gui.av_frame import AudioVideoFrame
from src.libs.gui.video_frame import VideoFrame
from src.libs.gui.audio_frame import AudioFrame

class MainFrame(ctk.CTkFrame):
    """Main Page of the GUI"""
    def __init__(self, parent, controller):
        """
        Initialize the main page

        Args:
            parent (tk.Frame): Parent frame
            controller (App): Controller class for the GUI
        """
        ctk.CTkFrame.__init__(self, parent)
        self.controller = controller
        self.controller.grid_columnconfigure(0, weight=1)
        
        self.icon = ctk.CTkImage(
            light_image=Image.open("src/data/images/static/driver.png"),
            dark_image=Image.open("src/data/images/static/driver.png"),
            size=(200, 200)
            )
        
        self.image = ctk.CTkLabel(self, image=self.icon, text="")
        self.image.pack(pady=15, padx=10)
        self.image.place(relx=0.5, rely=0.30, anchor='center')

        self.label2 = ctk.CTkLabel(self, text="Choose the recording mode:")
        self.label2.pack(pady=20, padx=10)
        self.label2.place(relx=0.5, rely=0.52, anchor='center')

        self.button = ctk.CTkButton(self, text="Audio and Video", command=lambda: controller.show_frame(AudioVideoFrame.__name__))
        self.button.pack(pady=12, padx=10)
        self.button.place(relx=0.5, rely=0.62, anchor='center')
        
        self.button1 = ctk.CTkButton(self, text="Video Only", command=lambda: controller.show_frame(VideoFrame.__name__))
        self.button1.pack(pady=12, padx=10)
        self.button1.place(relx=0.5, rely=0.72, anchor='center')

        self.button2 = ctk.CTkButton(self, text="Audio Only", command=lambda: controller.show_frame(AudioFrame.__name__))
        self.button2.pack(pady=12, padx=10)
        self.button2.place(relx=0.5, rely=0.82, anchor='center')