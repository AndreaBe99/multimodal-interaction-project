import customtkinter as ctk
import sys 
sys.path.append('./')

from src.libs.gui.main_frame import MainFrame
from src.libs.gui.av_frame import AudioVideoFrame
from src.libs.gui.video_frame import VideoFrame
from src.libs.gui.audio_frame import AudioFrame

class App(ctk.CTk):
    """Controller class for the GUI"""
    
    def __init__(self):
        ctk.CTk.__init__(self)
        self.title("Driver Monitoring System")
        self.geometry("600x600")

        self.container = ctk.CTkFrame(self)
        self.container.pack(fill=ctk.BOTH, expand=True)
        self.container.grid_rowconfigure(0, weight=1)
        self.container.grid_columnconfigure(0, weight=1)
        
        self.frames = {}
        for F in (MainFrame, AudioVideoFrame, VideoFrame, AudioFrame):
            page_name = F.__name__
            frame = F(parent=self.container, controller=self)
            self.frames[page_name] = frame
            frame.grid(row=0, column=0, sticky="nsew")
        self.show_frame(MainFrame.__name__)

    def show_frame(self, page_name):
        """
        Show a frame for the given page name

        Args:
            page_name (str): Name of the page to be shown
        """
        frame = self.frames[page_name]
        frame.tkraise()
        
        
if __name__ == "__main__":
    app = App()
    app.mainloop()